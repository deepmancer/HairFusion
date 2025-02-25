import os
from os.path import join as opj
from typing import Any, Optional
import omegaconf

import cv2
import einops
import torch
import torch as th
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    normalization
)

from einops import rearrange, repeat
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock, Upsample
from ldm.util import exists

# resolution 16 ~ 64
class WarpingUNetCA10(UNetModel):
    def __init__(
        self,
        dim_head_denorm=1,
        added_pose=None,
        added_pose_channels_out=None,
        align_hint_channels=3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        warp_flow_blks = []
        warp_zero_convs = []

        self.added_pose = added_pose
        if self.added_pose is not None:
            self.added_pose_channels = 3 if len(self.added_pose) ==1 else 6
        else:
            self.added_pose_channels = None
        self.added_pose_channels_out = added_pose_channels_out
        self.align_hint_channels = align_hint_channels
        self.encode_output_chs = [
            320,
            320,
            640,
            640,
            640,
            1280, 
            1280, 
            1280, 
            1280
        ]

        self.encode_output_chs2 = [
            320,
            320,
            320,
            320,
            640, 
            640, 
            640,
            1280, 
            1280
        ]

        if self.added_pose is not None:
            # noise_channels = 4
            add_pose_embed = [conv_nd(2, self.added_pose_channels, 16, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(2, 16, 16, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(2, 16, 32, 3, padding=1, stride=2), # 256
                    nn.SiLU(),
                    conv_nd(2, 32, 32, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(2, 32, 96, 3, padding=1, stride=2), # 128
                    nn.SiLU(),
                    conv_nd(2, 96, 96, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(2, 96, 256, 3, padding=1, stride=2), # 64
                    nn.SiLU(),
                    ]
            self.add_pose_embed = nn.Sequential(*add_pose_embed)
            add_pose_proj = [
                zero_module(conv_nd(2, 256, self.added_pose_channels_out, 1, padding=0))
            ]
            self.add_pose_proj = nn.Sequential(*add_pose_proj)


        if self.align:
            pose_proj = []
            pose_embed = [conv_nd(2, self.align_hint_channels, 16, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(2, 16, 16, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(2, 16, 32, 3, padding=1, stride=2), # 256
                    nn.SiLU(),
                    conv_nd(2, 32, 32, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(2, 32, 96, 3, padding=1, stride=2), # 128
                    nn.SiLU(),
                    conv_nd(2, 96, 96, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(2, 96, 256, 3, padding=1, stride=2), # 64
                    nn.SiLU(),
                    ]
            self.pose_embed = nn.Sequential(*pose_embed)
        
        for in_ch, cont_ch in zip(self.encode_output_chs, self.encode_output_chs2):
            dim_head = in_ch // self.num_heads
            dim_head = dim_head // dim_head_denorm
            warp_flow_blks.append(SpatialTransformer(
                in_channels=in_ch,
                n_heads=self.num_heads,
                d_head=dim_head,
                depth=self.transformer_depth,
                context_dim=cont_ch,
                use_linear=self.use_linear_in_transformer,
                use_checkpoint=self.use_checkpoint,
            ))
            warp_zero_convs.append(self.make_zero_conv(in_ch))
            if self.align:
                pose_proj.append(zero_module(conv_nd(2, 256, in_ch, 1, padding=0)))
        self.warp_flow_blks = nn.ModuleList(reversed(warp_flow_blks))
        self.warp_zero_convs = nn.ModuleList(reversed(warp_zero_convs))
        if self.align:
            self.pose_proj = nn.ModuleList(reversed(pose_proj))
        self.attention_store = {}
        self.input_path = None

    def make_zero_conv(self, channels):
        return zero_module(conv_nd(2, channels, channels, 1, padding=0))
    def forward(self, x, timesteps=None, context=None, control=None, pose_input=None, add_pose_input=None, only_mid_control=False, **kwargs):
        hs = []

        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:                 
            hint = control.pop()


        if add_pose_input is not None:
            pose_embedding = self.add_pose_embed(add_pose_input)
            _, _, h_h, h_w = h.shape
            pose_embedding = F.interpolate(pose_embedding, (h_h, h_w))  # resize
            h += self.add_pose_proj[0](pose_embedding)

        # resolution 8 is skipped
        for module in self.output_blocks[:3]:
            control.pop()
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        pose=None
        if self.align:
            # b = x.shape[0]
            pose_qk = torch.cat(pose_input) # 2B, 3, 512, 512
            pose = self.pose_embed(pose_qk) # 2B, 256 64 64

        n_warp = len(self.encode_output_chs)
        for i, (module, warp_blk, warp_zc) in enumerate(zip(self.output_blocks[3:n_warp+3], self.warp_flow_blks, self.warp_zero_convs)):
            if control is None or (h.shape[-2] == 8 and h.shape[-1] == 6):
                assert 0, f"shape is wrong : {h.shape}"
            else:
                hint = control.pop()
                pose_zc = self.pose_proj[i] if self.align else None
                h = self.warp(h, hint, warp_blk, warp_zc, pose_zc, pose = pose, step = [timesteps[0].item(),i, self.input_path])
                h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        for module in self.output_blocks[n_warp+3:]:
            if control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        return self.out(h)
    def warp(self, x, hint, crossattn_layer, zero_conv, pose_zc=None, pose=None, mask1=None, mask2=None, step=None):
        hint = rearrange(hint, "b c h w -> b (h w) c").contiguous()
        crossattn_layer.attention_store = None
        crossattn_layer.step = step
        num_step, idx, _ = step
        if self.align:
            _,_,h,w = x.shape
            pose = F.interpolate(pose, (h,w)) # resize
            pose = pose_zc(pose) # 2B, 256 64 64 -> 2B, 1280 16 16
            output = crossattn_layer(x, hint, pose) # SpatialTransformer
        else:
            output = crossattn_layer(x, hint)
        if crossattn_layer.attention_store is not None:
            self.attention_store[idx] = crossattn_layer.attention_store
        crossattn_layer.step = None
        crossattn_layer.attention_store = None
        output = zero_conv(output)
        return output + x


class NoZeroConvControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            use_VAEdownsample=False,
            cond_first_ch=8,
            # reg decoder
            use_regdecoder=False,
            out_channels=4,
            no_inputadd=False,
   ):
        self.use_regdecoder = use_regdecoder
        self.no_inputadd = no_inputadd
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received um_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.use_VAEdownsample = use_VAEdownsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.cond_first_block = TimestepEmbedSequential(
            zero_module(conv_nd(dims, cond_first_ch, model_channels, 3, padding=1))
        )


        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # regularizer
        if self.use_regdecoder:
            self.output_blocks = nn.ModuleList([])
            for level, mult in list(enumerate(channel_mult))[::-1]:
                for i in range(self.num_res_blocks[level]+1):
                    ich = input_block_chans.pop()
                    layers = [
                        ResBlock(
                            ch + ich,
                            time_embed_dim,
                            dropout,
                            out_channels=model_channels * mult,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                    ch = model_channels * mult
                    if ds in attention_resolutions:
                        if num_head_channels == -1:
                            dim_head = ch // num_heads
                        else:
                            num_heads = ch // num_head_channels
                            dim_head = num_head_channels
                        if legacy:
                            #num_heads = 1
                            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                        if exists(disable_self_attentions):
                            disabled_sa = disable_self_attentions[level]
                        else:
                            disabled_sa = False
                        if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                            layers.append(
                                AttentionBlock(
                                    ch,
                                    use_checkpoint=use_checkpoint,
                                    num_heads=num_heads_upsample,
                                    num_head_channels=dim_head,
                                    use_new_attention_order=use_new_attention_order,
                                ) if not use_spatial_transformer else SpatialTransformer(
                                    ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                    disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                    use_checkpoint=use_checkpoint
                                )
                            )
                    if level and i == self.num_res_blocks[level]:
                        out_ch = ch
                        layers.append(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                            )
                            if resblock_updown
                            else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                        )
                        ds //= 2
                    self.output_blocks.append(TimestepEmbedSequential(*layers))
                    self._feature_size += ch
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
            )

    def forward(self, x, hint, timesteps, context, only_mid_control=False, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if not self.use_VAEdownsample:
            guided_hint = self.input_hint_block(hint, emb, context)
        else:
            guided_hint = self.cond_first_block(hint, emb, context)

        outs = []
        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            if guided_hint is not None:
                if self.no_inputadd:
                    h = guided_hint
                else:
                    h = module(h, emb, context)
                    h += guided_hint
                hs.append(h)
                guided_hint = None
            else:                                                
                h = module(h, emb, context)
                hs.append(h)
            outs.append(h)

        h = self.middle_block(h, emb, context)
        outs.append(h)
        if self.use_regdecoder:
            h = h + outs[-1]
            for i, (module, control_feat) in enumerate(zip(self.output_blocks, outs[:-1][::-1])):
                if only_mid_control:
                    h = torch.cat([h, hs.pop()], dim=1)
                else:
                    h = torch.cat([h, hs.pop() + control_feat], dim=1)
                h = module(h, emb, context)
            h = h.type(x.dtype)
            cond_out = self.out(h)
            return outs, cond_out
        else:
            return outs, None