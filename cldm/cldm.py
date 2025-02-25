import os
from os.path import join as opj
from typing import Any, Optional
import omegaconf
from glob import glob

import cv2
import einops
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch as th
import torch.nn as nn
from cleanfid import fid
from pytorch_lightning.utilities.distributed import rank_zero_only
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from eval_models import PerceptualLoss

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    normalization
)

from utils import tensor2img, resize_mask
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock, Upsample
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        if self.no_control: control = None
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
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)

class ControlNet(nn.Module):
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
            no_inputadd=False
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
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
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
        self.cond_first_ch = cond_first_ch

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
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])
        
        
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        if self.use_VAEdownsample:
            for p in self.input_hint_block.parameters():
                p.requires_grad = False

        self.cond_first_block = TimestepEmbedSequential(
            zero_module(conv_nd(dims, cond_first_ch, model_channels, 3, padding=1))  # input_hint_block 마지막이랑 똑같이 맞추자.
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
                self.zero_convs.append(self.make_zero_conv(ch))
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
                self.zero_convs.append(self.make_zero_conv(ch))
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
        self.middle_block_out = self.make_zero_conv(ch)
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
        #############
    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

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
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                if self.no_inputadd:
                    h = guided_hint
                else:
                    h += guided_hint
                guided_hint = None
            else:                                                
                h = module(h, emb, context)
            hs.append(h)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))
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

class ControlLDM(LatentDiffusion):
    def __init__(
            self, 
            control_stage_config, 
            validation_config, 
            control_key, 
            only_mid_control, 
            use_VAEdownsample=False,
            use_regdecoder=False,
            use_noisy_cond=False,
            all_unlocked=False,
            config_name="",
            control_scales=None,
            use_pbe_weight=False,
            u_cond_percent=0.0,
            img_H=512,
            img_W=384,
            imageclip_trainable=True,
            use_u_cond_cnet=False,
            u_cond_cnet_percent=0.0,
            use_custom_cond_stage_key=False,
            pbe_train_mode=False,
            always_learnable_param=False,
            align_pose='dp',
            *args, 
            **kwargs
        ):
        self.control_stage_config = control_stage_config
        self.use_pbe_weight = use_pbe_weight
        self.u_cond_percent = u_cond_percent
        self.img_H = img_H
        self.img_W = img_W
        self.config_name = config_name
        self.imageclip_trainable = imageclip_trainable
        self.use_u_cond_cnet = use_u_cond_cnet
        self.u_cond_cnet_percent = u_cond_cnet_percent
        self.use_custom_cond_stage_key = use_custom_cond_stage_key
        self.pbe_train_mode = pbe_train_mode
        self.always_learnable_param = always_learnable_param
        super().__init__(*args, **kwargs)
        control_stage_config.params["use_VAEdownsample"] = use_VAEdownsample
        control_stage_config.params["use_regdecoder"] = use_regdecoder
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        if control_scales is None:
            self.control_scales = [1.0] * 13
        else:
            self.control_scales = control_scales
        self.first_stage_key_cond = kwargs.get("first_stage_key_cond", None)
        self.valid_config = validation_config
        self.use_VAEDownsample = use_VAEdownsample
        self.use_regdecoder = use_regdecoder
        self.use_noisy_cond = use_noisy_cond
        self.all_unlocked = all_unlocked
        self.align_pose = align_pose
        
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        if isinstance(self.control_key, omegaconf.listconfig.ListConfig):
            control_lst = []
            for key in self.control_key:
                control = batch[key]
                if bs is not None:
                    control = control[:bs]
                control = control.to(self.device)
                control = einops.rearrange(control, 'b h w c -> b c h w')
                control = control.to(memory_format=torch.contiguous_format).float()
                control_lst.append(control)
            control = control_lst
        else:
            control = batch[self.control_key]
            if bs is not None:
                control = control[:bs]
            control = control.to(self.device)
            control = einops.rearrange(control, 'b h w c -> b c h w')
            control = control.to(memory_format=torch.contiguous_format).float()
            control = [control]
        cond_dict = dict(c_crossattn=[c], c_concat=control)  # cross attn은 imageclip이면 224x224 이미지
        if self.first_stage_key_cond is not None:
            first_stage_cond = []
            for key in self.first_stage_key_cond:
                if not "mask" in key:
                    cond, _ = super().get_input(batch, key, *args, **kwargs)
                else:
                    cond, _ = super().get_input(batch, key, no_latent=True, *args, **kwargs)      
                first_stage_cond.append(cond)
            first_stage_cond = torch.cat(first_stage_cond, dim=1)
            cond_dict["first_stage_cond"] = first_stage_cond
        return x, cond_dict

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)       
        
        cond_output_dict = None
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond["c_crossattn"], 1)
        if self.proj_out is not None:
            if cond_txt.shape[-1] == 1024:
                cond_txt = self.proj_out(cond_txt)  # [BS x 1 x 768]
        if self.always_learnable_param:
            cond_txt = self.get_unconditional_conditioning(cond_txt.shape[0])

        add_pose_input = None
        if diffusion_model.added_pose is not None: # pose embedding added to latent noise
            pose_nth = einops.rearrange(self.batch['image_nth'], 'b h w c -> b c h w').contiguous() # src
            pose_dp = einops.rearrange(self.batch['image_keypoints'], 'b h w c -> b c h w').contiguous() # hair
            if len(diffusion_model.added_pose) == 2:
                add_pose_input = [pose_nth, pose_dp]  # query (src), key (trg)
                add_pose_input = torch.cat(add_pose_input, dim=1)  # B, 2*3, 512, 512
            else:
                if diffusion_model.added_pose == ['nth']:
                    add_pose_input = pose_nth
                elif diffusion_model.added_pose == ['dp']:
                    add_pose_input = pose_dp
                else:
                    raise NotImplementedError


        cond_output = None
        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            if "first_stage_cond" in cond:
                x_noisy = torch.cat([x_noisy, cond["first_stage_cond"]], dim=1)
            if not self.use_VAEDownsample:
                hint = cond["c_concat"]
            else:
                hint = []
                for h in cond["c_concat"]:
                    if h.shape[2] == self.img_H and h.shape[3] == self.img_W:
                        h = self.encode_first_stage(h)
                        h = self.get_first_stage_encoding(h).detach()
                    hint.append(h)
            hint = torch.cat(hint, dim=1)
            if self.use_noisy_cond:
                cond_noise = torch.randn_like(hint)
                t1 = torch.ones((x_noisy.shape[0],), device=self.device).long()  #  TODO: t=[1,1]
                hint = self.q_sample(x_start=hint, t=t1, noise=cond_noise)  # noisy cond
            
            if self.use_custom_cond_stage_key and self.cond_stage_key is not None:
                hint = super(LatentDiffusion, self).get_input(self.batch, self.cond_stage_key)[:x_noisy.shape[0]]

            control, cond_output = self.control_model(x=x_noisy, hint=hint, timesteps=t, context=cond_txt, only_mid_control=self.only_mid_control) # false
            if len(control) == len(self.control_scales):
                control = [c * scale for c, scale in zip(control, self.control_scales)]
            
            mask1 = None
            mask2 = None

            if self.align_pose == 'nth':
                pose_q = einops.rearrange(self.batch['image_nth'], 'b h w c -> b c h w').contiguous()  # src
                pose_k = einops.rearrange(self.batch['hair_nth'], 'b h w c -> b c h w').contiguous()  # hair
            elif self.align_pose == 'dp':
                pose_q = einops.rearrange(self.batch['image_keypoints'], 'b h w c -> b c h w').contiguous() # src
                pose_k = einops.rearrange(self.batch['hair_keypoints'], 'b h w c -> b c h w').contiguous() # hair
            elif self.align_pose == 'both':
                pose_q1 = einops.rearrange(self.batch['image_nth'], 'b h w c -> b c h w').contiguous()  # src
                pose_q2 = einops.rearrange(self.batch['image_keypoints'], 'b h w c -> b c h w').contiguous() # src
                pose_q = torch.cat([pose_q1, pose_q2], dim=1)
                pose_k1 = einops.rearrange(self.batch['hair_nth'], 'b h w c -> b c h w').contiguous()  # hair
                pose_k2 = einops.rearrange(self.batch['hair_keypoints'], 'b h w c -> b c h w').contiguous() # hair
                pose_k = torch.cat([pose_k1, pose_k2], dim=1)
            else:
                raise NotImplementedError

            pose_input = [pose_q, pose_k] # query (src), key (trg)
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, pose_input=pose_input,
                                  add_pose_input=add_pose_input, only_mid_control=self.only_mid_control, mask1=mask1, mask2=mask2)
        if cond_output is not None:
            if cond_output_dict is None:
                cond_output_dict = dict()
            cond_output_dict["cond_output"] = cond_output
            cond_output_dict["cond_input"] = hint
            if self.use_noisy_cond:
                cond_output_dict["cond_noise"] = cond_noise
        return eps, cond_output_dict
    
    @torch.no_grad()
    def mask_resize(self, m, h, w, inverse=False):
        m = F.interpolate(m, (h, w), mode="nearest")
        if inverse:
            m = 1-m
        return m
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        if not self.kwargs["use_imageCLIP"]:
            return self.get_learned_conditioning([""] * N)
        else:
            return self.learnable_vector.repeat(N,1,1)
    @torch.no_grad()
    def get_unconditional_conditioning_cnet(self, N):
        return self.learnable_matrix.repeat(N,1,1,1)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        if self.first_stage_key_cond:
            first_stage_cond = c["first_stage_cond"][:N]
            for key_idx, key in enumerate(self.first_stage_key_cond):
                cond = batch[key]
                if len(cond.shape) == 3:
                    cond = cond[..., None]
                cond = rearrange(cond, "b h w c -> b c h w")
                cond = cond.to(memory_format=torch.contiguous_format).float()
                log[f"first_stage_cond_{key_idx}"] = cond
        c_cat = [i[:N] for i in c["c_concat"]]
        c = c["c_crossattn"][0][:N]
        if c.ndim == 4:
            c = self.get_learned_conditioning(c)
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        
        x = batch[self.first_stage_key]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        log["input"] = x
        log["reconstruction"] = self.decode_first_stage(z)
        log_c_cat = torch.cat(c_cat, dim=1)
        if torch.all(log_c_cat >= 0):
            log["control"] = log_c_cat * 2.0 - 1.0  
        else:
            log["control"] = log_c_cat
        if not self.kwargs["use_imageCLIP"]:
            log["clip conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], 
            size=16)
        else:
            x = batch[self.cond_stage_key]
            if len(x.shape) == 3:
                x = x[..., None]
            x = rearrange(x, 'b h w c -> b c h w')
            x = x.to(memory_format=torch.contiguous_format).float()
            log["clip conditioning"] = x

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid
        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": c_cat, "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid
        if unconditional_guidance_scale >= 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            if self.use_u_cond_cnet:
                uc_cat = [self.get_unconditional_conditioning_cnet(N)]
            else:
                uc_cat = c_cat  # torch.zeros_like(c_cat)
            if self.u_cond_percent == 1:
                print(f"log images c=uc_cross")
                c = uc_cross
            cond = {"c_concat": c_cat, "c_crossattn": [c]}
            uc_full = {"c_concat": uc_cat, "c_crossattn": [uc_cross]}
            if self.first_stage_key_cond:
                cond["first_stage_cond"] = first_stage_cond
                uc_full["first_stage_cond"] = first_stage_cond
            samples_cfg, _, cond_output_dict = self.sample_log(cond=cond,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
            if cond_output_dict is not None:
                cond_sample = cond_output_dict["cond_sample"]             
                cond_sample = self.decode_first_stage(cond_sample)
                log[f"cond_sample"] = cond_sample        

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps=5, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates, cond_output_dict = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates, cond_output_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        print("=====configure optimizer=====")
        if self.pbe_train_mode:
            print("pbe train mode")
            params = list(self.model.parameters())
            print("- unet is added")
            params += list(self.cond_stage_model.final_ln.parameters())
            print("- cond stage model final ln is added")
            params += list(self.cond_stage_model.mapper.parameters())
            print("- cond stage model mapper is added")
            params += list(self.proj_out.parameters())
            print("- proj_out layer is added")
            params.append(self.learnable_vector)
            print("- learnable vector is added")
            opt = torch.optim.AdamW(params, lr=lr)
            print("============================")
            return opt
        params = list(self.control_model.parameters())
        print("control model is added")
        if self.all_unlocked:
            params += list(self.model.parameters())
            print("Unet is added")
        else:
            if not self.sd_locked:
                params += list(self.model.diffusion_model.output_blocks.parameters())
                print("Unet output blocks is added")
                params += list(self.model.diffusion_model.out.parameters())
                print("Unet out is added")
            if "pbe" in self.config_name:
                if self.unet_config.params.in_channels != 9:
                    params += list(self.model.diffusion_model.input_blocks[0].parameters())
                    print("Unet input block is added")
            else:
                if self.unet_config.params.in_channels != 4:
                    params += list(self.model.diffusion_model.input_blocks[0].parameters())
                    print("Unet input block is added")
        if self.cond_stage_trainable:
            if hasattr(self.cond_stage_model, "final_ln"):
                params += list(self.cond_stage_model.final_ln.parameters())
                print("cond stage model final ln is added")
            if hasattr(self.cond_stage_model, "mapper"):
                params += list(self.cond_stage_model.mapper.parameters())
                print("cond stage model mapper is added")
        if self.proj_out is not None:
            params += list(self.proj_out.parameters())
            print("proj out is added")
        if self.learnable_vector is not None:
            params.append(self.learnable_vector)
            print("learnable vector is added")
        if hasattr(self.model.diffusion_model, "warp_flow_blks"):
            params += list(self.model.diffusion_model.warp_flow_blks.parameters())
            print(f"warp flow blks is added")
        if hasattr(self.model.diffusion_model, "warp_zero_convs"):
            params += list(self.model.diffusion_model.warp_zero_convs.parameters())
            print(f"warp zero convs is added")
        if hasattr(self.model.diffusion_model, "pose_embed"):
            params += list(self.model.diffusion_model.pose_embed.parameters())
            print(f"pose embed is added")
        if hasattr(self.model.diffusion_model, "pose_proj"):
            params += list(self.model.diffusion_model.pose_proj.parameters())
            print(f"pose proj is added")
        if self.learnable_matrix is not None:
            params.append(self.learnable_matrix)
            print("learnable matrix is added")
        opt = torch.optim.AdamW(params, lr=lr)
        print("============================")
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
        
    @torch.no_grad()
    def on_validation_epoch_start(self):
        self.ddim_sampler = DDIMSampler(self)
        self.validation_gene_dirs = []
        for data_type in ["pair", "unpair"]:
            to_dir = opj(self.valid_config.img_save_dir, f"{data_type}_{self.current_epoch}")
            self.validation_gene_dirs.append(to_dir)
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        pass
        
        # if batch_idx > 2: return 
        # self.batch = batch
        # data_type = "pair" if dataloader_idx==0 else "unpair"
        # z, c = self.get_input(batch, self.first_stage_key)
        # x_recon = self.decode_first_stage(z)
        # shape = (4, self.img_H//8, self.img_W//8)
        # bs = z.shape[0]
        # c_crossattn = c["c_crossattn"][0][:bs]
        # if c_crossattn.ndim == 4:
        #     c_crossattn = self.get_learned_conditioning(c_crossattn)
        #     c["c_crossattn"] = [c_crossattn]
        # uc_cross = self.get_unconditional_conditioning(bs)
        # if self.use_u_cond_cnet:
        #     uc_cat = [self.get_unconditional_conditioning_cnet(bs)]
        # else:
        #     uc_cat = c["c_concat"]
        # uc_full = {"c_concat": uc_cat, "c_crossattn": [uc_cross]}
        # uc_full["first_stage_cond"] = c["first_stage_cond"]

        # samples, intermediates, _ = self.ddim_sampler.sample(
        #     self.valid_config.ddim_steps,
        #     bs,
        #     shape,
        #     c,
        #     x_T=None,
        #     verbose=False,
        #     eta=self.valid_config.eta,
        #     mask=None,
        #     x0=None,
        #     unconditional_guidance_scale=5.0,
        #     unconditional_conditioning=uc_full
        # )
        # x_samples = self.decode_first_stage(samples)
        # to_dir = opj(self.valid_config.img_save_dir, f"{data_type}_{self.current_epoch}")
        # os.makedirs(to_dir, exist_ok=True)
        # for x_sample, cloth, w_cloth, gt, fn, recon in zip(x_samples, batch["cloth"], batch["cloth_warped"], batch["image"], batch["img_fn"], x_recon):
        #     x_sample_img = tensor2img(x_sample)
        #     x_recon_img = tensor2img(recon)
        #     cloth_img = np.uint8((cloth.detach().cpu()+1)/2 * 255.0)
        #     w_cloth_img = np.uint8((w_cloth.detach().cpu()+1)/2 * 255.0)
        #     gt_img = np.uint8((gt.detach().cpu()+1)/2 * 255.0)
        #     cloth_save = np.concatenate([x_sample_img, gt_img, cloth_img, w_cloth_img, x_recon_img], axis=1)

        #     to_path = opj(to_dir, fn)
        #     cv2.imwrite(to_path, cloth_save[:,:,::-1])

