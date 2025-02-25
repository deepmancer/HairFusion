from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any
import cv2
import numpy as np

from ldm.modules.diffusionmodules.util import checkpoint

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_
@torch.no_grad()
def attn_mask_resize(m,h,w):
    """
    m : [BS x 1 x mask_h x mask_w] => downsample, reshape and bool, [BS x h x w]
    """  
    m = F.interpolate(m, (h, w)).squeeze(1).contiguous()
    m = torch.where(m>=0.5, True, False)
    return m

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.step = None
        self.timestep_list_raw = [981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741, 721, 701,
                             681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461, 441, 421, 401,
                             381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181, 161, 141, 121, 101,
                             81, 61, 41, 21, 1]
        self.attention_store = None

    def forward(self, x, context=None, mask=None, mask1=None, mask2=None, use_attention_loss=False, use_attention_tv_loss=False, pose=None):
        h = self.heads
        is_self_attn = context is None
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape # 찍어보기
        if pose is not None:
            pose = rearrange(pose, 'b c h w -> b (h w) c').contiguous()  # 2B, 256, 1280
            pose = pose[:,None].repeat(1, h, 1, 1)  # 2B, 8, 256, 1280
            pose_q, pose_k = pose[:b], pose[b:] # B, 4096, 320 / B, 4096, 320
            q += pose_q # src
            k += pose_k # hair

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
        attn_mask = None
        if exists(mask1) or exists(mask2):  # [BS x 1 x H x W] float
            if mask1.ndim == 4 and mask2.ndim == 4:
                _, HW, hw = sim.shape
                bs = mask1.shape[0]
                dx = int((HW//12) ** 0.5)
                mH = int(4*dx)
                mW = int(3*dx)
                dx = int((hw//12) ** 0.5)
                mh = int(4*dx)
                mw = int(3*dx)
                if mH != 8:
                    mask1 = attn_mask_resize(mask1, mH, mW)  # [BS x H x W]
                    mask2 = attn_mask_resize(mask2, mh, mw)  # [BS x h x w]
                    
                    attn_mask = mask1.reshape(bs, -1).unsqueeze(-1) * mask2.reshape(bs, -1).unsqueeze(1)  # [BS x HW x hw]               
                    attn_mask = repeat(attn_mask, "b HW hw -> (b h) HW hw", h=h)
        
                    assert attn_mask.shape == sim.shape, f"mask : {attn_mask.shape}, attn map : {sim.shape}"   
                             
                    if not (use_attention_loss or use_attention_tv_loss):
                        max_neg_value = -torch.finfo(sim.dtype).max
                        sim.masked_fill_(attn_mask, max_neg_value)
                        
            else:
                raise NotImplementedError
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
               
        # temperature
        # temperature = 4
        # sim = sim / temperature
        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)  # [(BSxh) x HW x hw]

        attn_loss = torch.tensor(0, dtype=x.dtype, device=x.device)
        if use_attention_loss:
            if attn_mask is not None:
                attn_mask_for_loss = mask1.reshape(bs, -1).unsqueeze(-1).float() * (1 - mask2.reshape(bs, -1).unsqueeze(1).float())  # agn의 cloth만 1
                masked_attn = (sim * attn_mask_for_loss).sum(dim=-1)
                mask1_repeat = repeat(mask1, "b h w -> (b n) (h w)", n=h)
                attn_loss = F.mse_loss(masked_attn.float(), mask1_repeat.float())
                
        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        #### remove 
        if attn_mask is not None:
            def minmax(x):
                mi, _ = torch.min(x, dim=-1, keepdim=True)
                ma, _ = torch.max(x, dim=-1, keepdim=True)
                return (x - mi) / (ma - mi)
            def mult255(x):
                return x
            def max_(x):
                HW, hw = x.shape
                _, index = torch.max(x, dim=-1)
                zeros = torch.zeros_like(x)
                zeros[torch.arange(len(x)), index] = 1
                return zeros 
            import cv2
            import numpy as np
            import os
            from os.path import join as opj

        if not (use_attention_loss or use_attention_tv_loss):
            return self.to_out(out)
        else:
            return self.to_out(out), attn_loss

class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None
        self.step = None
        self.scale = dim_head ** -0.5
        self.timestep_list_raw = [981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741, 721, 701,
                             681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461, 441, 421, 401,
                             381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181, 161, 141, 121, 101,
                             81, 61, 41, 21, 1]
        self.attention_store = None

    def forward(self, x, context=None, mask=None, pose=None):
        q = self.to_q(x)
        is_self_attn = context is None
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        if pose is not None:
            pose = rearrange(pose, 'b c h w -> b (h w) c').contiguous()  # 2B, 256, 1280
            pose_q, pose_k = pose[:b], pose[b:] # B, 4096, 320 / B, 4096, 320
            q += pose_q # src
            k += pose_k # hair

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )


        if (not is_self_attn) and exists(self.step):
            step, idx, _ = self.step # step 은 25 이후, idx는 012 345 678 -> 67만
            if idx  ==6 or idx ==7:
                if _ATTN_PRECISION == "fp32":
                    with torch.autocast(enabled=False, device_type='cuda'):
                        q, k = q.float(), k.float()
                        sim_tmp = einsum('b i d, b j d -> b i j', q, k) * self.scale  # head, 256, 256
                else:
                    sim_tmp = einsum('b i d, b j d -> b i j', q, k) * self.scale
                sim_tmp = sim_tmp.reshape(b, self.heads, sim_tmp.shape[-2], sim_tmp.shape[-2])
                sim_tmp = torch.mean(sim_tmp, dim=1)
                self.attention_store = sim_tmp

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)        

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, pose=None):
        if pose is not None:
            return checkpoint(self._forward, (x, context, pose), self.parameters(), self.checkpoint)
        else:
            return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, pose=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context, pose=pose) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear
        self.step = None
        self.attention_store = None

    def forward(self, x, context=None, pose=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if self.step is not None:
                block.attn1.step = self.step
                block.attn2.step = self.step
            x = block(x, context=context[i], pose=pose)
            block.attn1.step = None
            block.attn2.step = None
            if block.attn2.attention_store is not None:
                self.attention_store = block.attn2.attention_store
                block.attn2.attention_store = None
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


