import os
from os.path import join as opj
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from importlib import import_module
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from einops import rearrange

from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from utils import tensor2img

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type=str)
parser.add_argument("--model_load_path", type=str)
parser.add_argument("--vae_load_path", type=str, default="./models/realisticVisionV51_v51VAE.ckpt", help="None is sd vae")
parser.add_argument("--save_name", type=str, default="dummy")
parser.add_argument("--batch_size", type=int, default=8)  # VRAM 36GB
parser.add_argument("--scale", type=float, default=5.0)

parser.add_argument("--data_root_dir", type=str, default="data/test_data/")
parser.add_argument("--n_iters", type=int, default=None)

parser.add_argument("--save_inter", action="store_true")
parser.add_argument("--log_every_t", type=int, default=100)
parser.add_argument("--save_root_dir", type=str, default=None)

parser.add_argument("--ddim_steps", type=int, default=50)
parser.add_argument("--img_H", type=int, default=512)
parser.add_argument("--img_W", type=int, default=512)
parser.add_argument("--eta", type=float, default=0.0)
parser.add_argument("--default_prompt", type=str, default="")
parser.add_argument("--n_prompt", type=str, default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
parser.add_argument("--resampling_trick", default=False)
parser.add_argument("--resampling_trick_repeat", type=int, default=10)
parser.add_argument("--infer_idx", type=int, default=None)
parser.add_argument("--total_idx", type=int, default=None)

parser.add_argument("--last_n_blend", type=float, default=None, help="adaptive blending for last n timesteps during infererence")


args = parser.parse_args()

config_name = args.config_name
model_load_path = args.model_load_path
vae_load_path = args.vae_load_path

def build_config(args, config_path=None):
    if config_path is None:
        config_path = args.config_path
    config = OmegaConf.load(config_path)
    config.model.params.setdefault("use_VAEdownsample", False)
    config.model.params.setdefault("use_imageCLIP", False)
    config.model.params.setdefault("use_lastzc", False)
    config.model.params.setdefault("use_regdecoder", False)
    config.model.params.setdefault("use_pbe_weight", False)
    config.model.params.setdefault("u_cond_percent", 0.0)
    if args is not None:
        for k, v in vars(args).items():
            config.model.params.setdefault(k, v)
    if not config.model.params.get("validation_config", None):
        config.model.params.validation_config = OmegaConf.create()
    config.model.params.validation_config.ddim_steps = config.model.params.validation_config.get("ddim_steps", 50)
    config.model.params.validation_config.eta = config.model.params.validation_config.get("eta", 0.0)
    config.model.params.validation_config.scale = config.model.params.validation_config.get("scale", 1.0)
    if args is not None:
        config.model.params.validation_config.img_save_dir = args.valid_img_save_dir
        config.model.params.validation_config.real_dir = args.valid_real_dir
    return config

with torch.no_grad():
    ddim_steps = args.ddim_steps
    batch_size = args.batch_size
    img_H = args.img_H
    img_W = args.img_W
    scale = args.scale
    eta = args.eta
    default_prompt = args.default_prompt
    n_prompt = args.n_prompt


    if args.config_name is None:
        config_path = opj("/".join(args.model_load_path.split("/")[:2]), "config.yaml")
    else:
        config_path = f"./configs/{args.config_name}.yaml"
    config = build_config(None, config_path)
    config_load = build_config(None, opj("/".join(args.model_load_path.split("/")[:2]), "config.yaml"))
    config.model.params.img_H = args.img_H
    config.model.params.img_W = args.img_W
    print(config)
    params = config.model.params
    params_load = config_load.model.params

    model = create_model(config_path=config_path, config=config)
    model.load_state_dict(load_state_dict(model_load_path, location="cpu"))
    if vae_load_path is not None:
        state_dict = load_state_dict(vae_load_path, location="cpu")
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'realistic' in args.vae_load_path:
                if "first_stage_model." in k and "loss." not in k:
                    new_k = k.replace("first_stage_model.", "")
                    new_state_dict[new_k] = v.clone()
            else:
                if "loss." not in k:
                    new_state_dict[k] = v.clone()
        model.first_stage_model.load_state_dict(new_state_dict)

    model = model.cuda()
    model.eval()
    ddim_sampler = DDIMSampler(
        model,
        resampling_trick=args.resampling_trick,
        last_n_blend=args.last_n_blend,
        resampling_trick_repeat=args.resampling_trick_repeat
    )

    if args.save_root_dir is None:
        to_dir = opj("output", args.save_name)
    else:
        to_dir = opj(args.save_root_dir, "output", args.save_name)

    os.makedirs(to_dir, exist_ok=True)

    dataset = getattr(import_module("dataset"), config.dataset_name)(
        args=args,
        data_root_dir=args.data_root_dir,
        img_H=img_H,
        img_W=img_W,
        default_prompt=default_prompt,
        is_test=True,
    )
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=batch_size, pin_memory=True)

    shape = (4, img_H//8, img_W//8)
    if args.infer_idx is not None and args.total_idx is not None:
        skip_block = len(dataloader) // args.total_idx + 1
        start_idx = skip_block * (args.infer_idx - 1)
        end_idx = skip_block * args.infer_idx

    idx = 0
    for batch_idx, batch in enumerate(dataloader):
        print(f"{batch_idx}/{len(dataloader)}")
        if (args.infer_idx is not None) and (args.total_idx is not None) and not (start_idx <= batch_idx < end_idx):
            print(f"skip {batch_idx}/{len(dataloader)}")
            continue

        z, c = model.get_input(batch, params.first_stage_key)
        x_recon = model.decode_first_stage(z)
        bs = z.shape[0]
        c_crossattn = c["c_crossattn"][0][:bs]
        if c_crossattn.ndim == 4:
            c_crossattn = model.get_learned_conditioning(c_crossattn)
            c["c_crossattn"] = [c_crossattn]

        mask = batch["agn_mask"]
        x0, _ = model.get_input(batch, params.first_stage_key)
        if len(mask.shape) == 3:
            mask = mask[..., None]
        mask = rearrange(mask, 'b h w c -> b c h w')
        mask = mask.to(memory_format=torch.contiguous_format).float()
        mask = mask.to(x0.device)
        mask = resize(mask, (img_H//8, img_W//8))


        uc_cross = model.get_unconditional_conditioning(bs)
        if config_load.model.params.get("use_c_cond_cnet", False):
            uc_cat  = [model.get_unconditional_conditioning_cnet(bs)]
        else:
            uc_cat = c["c_concat"]
        if config_load.model.params.get("u_cond_percent", 0) == 1:
            c["c_crossattn"] = [uc_cross]
        uc_full = {"c_concat": uc_cat, "c_crossattn": [uc_cross]}
        uc_full["first_stage_cond"] = c["first_stage_cond"]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        ddim_sampler.model.batch = batch
        samples, intermediates, _ = ddim_sampler.sample(
            ddim_steps,
            bs,
            shape,
            c,
            x_T=None,
            verbose=False,
            eta=eta,
            mask=mask,
            x0=x0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            log_every_t=args.log_every_t
        )

        x_samples = model.decode_first_stage(samples)

        for gt, ref_image, x_sample in zip(batch["image"], batch["ref_image"], x_samples):

            x_sample_img = tensor2img(x_sample)
            gt_img = np.uint8((gt.detach().cpu()+1)/2 * 255.0)
            ref_img = np.uint8((ref_image.detach().cpu()+1)/2 * 255.0)
        
            hair_save = np.concatenate([gt_img, ref_img, x_sample_img], axis=1)

            to_path = opj(to_dir, str(idx).zfill(4) + '.png')
            cv2.imwrite(to_path, x_sample_img[:,:,::-1])

            to_path_hair = opj(to_dir, str(idx).zfill(4) + '_full.png')
            cv2.imwrite(to_path_hair, hair_save[:,:,::-1])
            idx += 1

        if batch_idx+1 == args.n_iters:
            break
