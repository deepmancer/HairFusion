"""
HairFusion Full Inference Pipeline

This script performs end-to-end hair transfer inference using HairFusion.
Given a data directory with pairs.csv and aligned images, it:
1. Preprocesses images (face detection, cropping, landmark extraction, segmentation)
2. Extracts DensePose for spatial alignment
3. Generates agnostic images (hair region removal)
4. Runs the full hair transfer inference

Usage:
    python inference_full.py --data_dir /path/to/data

Expected data directory structure:
    data_dir/
        pairs.csv          # CSV with 'source_id' and 'target_id' columns
        aligned_image/     # Directory containing input images (source/target)
        
Output structure:
    data_dir/baselines/hairfusion/{target_id}_to_{source_id}/
        transferred.png    # Final hair-transferred image
"""

import os
import sys
import argparse
import random
import subprocess
import shutil
from pathlib import Path
from os.path import join as opj
from importlib import import_module

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from einops import rearrange
from kornia.morphology import dilation, erosion
from skimage import io, img_as_float32

# Add HairFusion to path
HAIRFUSION_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HAIRFUSION_DIR)

from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from utils import tensor2img, get_seg, get_seg_mask, get_nth, get_forehead, get_binary_from_img
from models.ffhq_dataset.landmarks_detector import LandmarksDetector
from models.face_parsing.model import BiSeNet


class PreprocessingPipeline:
    """Handles preprocessing of images: face detection, cropping, landmarks, segmentation."""
    
    def __init__(self, hairfusion_dir: str):
        self.hairfusion_dir = Path(hairfusion_dir)
        self.landmarks_model_path = self.hairfusion_dir / 'models' / 'shape_predictor_68_face_landmarks.dat'
        self.seg_model_path = self.hairfusion_dir / 'models' / 'face_segment16.pth'
        
        # Verify model files exist
        if not self.landmarks_model_path.exists():
            raise FileNotFoundError(f"Landmarks model not found: {self.landmarks_model_path}")
        if not self.seg_model_path.exists():
            raise FileNotFoundError(f"Segmentation model not found: {self.seg_model_path}")
        
        # Initialize models
        print("Loading preprocessing models...")
        self.landmarks_detector = LandmarksDetector(str(self.landmarks_model_path))
        
        self.seg_model = BiSeNet(n_classes=16)
        self.seg_model.cuda()
        self.seg_model.load_state_dict(torch.load(str(self.seg_model_path)))
        self.seg_model.eval()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.transform_resize = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        
        print("Preprocessing models loaded.")

    @staticmethod
    def get_crop_coords(keypoints, size, scale=2.5):
        """Get crop coordinates based on facial keypoints."""
        min_y, max_y = keypoints[:, 1].min(), keypoints[:, 1].max()
        min_x, max_x = keypoints[:, 0].min(), keypoints[:, 0].max()
        xc = (min_x + max_x) // 2
        yc = (min_y + max_y * 4) // 5
        h = w = min((max_x - min_x) * scale, min(size[0], size[1]))
        xc = min(max(0, xc - w // 2) + w, size[0]) - w // 2
        yc = min(max(0, yc - h // 2) + h, size[1]) - h // 2
        min_x, max_x = xc - w // 2, xc + w // 2
        min_y, max_y = yc - h // 2, yc + h // 2
        return int(min_x), int(min_y), int(max_x), int(max_y)

    def preprocess_image(self, img_path: Path, output_dirs: dict, crop_scale: float = 4.0) -> str:
        """
        Preprocess a single image: face detection, crop, landmarks, segmentation.
        Returns the filename of the preprocessed image (without directory).
        """
        raw_img_path = str(img_path)
        
        # Construct expected filename first (before expensive face detection)
        face_img_name = f'{img_path.stem}_01.png'
        
        # Check if already preprocessed - do this BEFORE face detection
        if all((output_dirs[k] / face_img_name.replace('.png', '.txt' if k == 'keypoints' else '.png')).exists() 
               for k in ['images', 'keypoints', 'mask_hair', 'mask_face', 'nth']):
            return face_img_name
        
        # Detect faces (only if not already preprocessed)
        detected_landmarks = self.landmarks_detector.get_landmarks(raw_img_path)
        if len(detected_landmarks) == 0:
            print(f"  WARNING: No faces detected in {img_path.name}, skipping...")
            return None
        
        # Use first detected face
        face_landmarks = detected_landmarks[0]
        crop_face_path = output_dirs['images'] / face_img_name
        
        # Load and crop image
        raw_img = img_as_float32(io.imread(raw_img_path))
        face_landmarks_np = np.array(face_landmarks)
        min_x, min_y, max_x, max_y = self.get_crop_coords(
            face_landmarks_np, (raw_img.shape[1], raw_img.shape[0]), scale=crop_scale
        )
        raw_img_cropped = raw_img[min_y:max_y, min_x:max_x]
        raw_img_cropped = (raw_img_cropped * 255).astype(np.uint8)
        raw_img_cropped_pil = Image.fromarray(raw_img_cropped)
        raw_img_cropped_pil.save(crop_face_path)
        
        # Transform landmarks to cropped coordinates
        face_landmarks_cropped = face_landmarks_np.copy()
        face_landmarks_cropped[:, 0] -= min_x
        face_landmarks_cropped[:, 1] -= min_y
        
        # Save keypoints
        keypoints_path = output_dirs['keypoints'] / face_img_name.replace('.png', '.txt')
        np.savetxt(keypoints_path, face_landmarks_cropped, fmt='%d', delimiter=',')
        
        # Load cropped image and scale keypoints
        image_raw = Image.open(crop_face_path).convert('RGB')
        img_size = image_raw.size[0]
        kp = np.loadtxt(keypoints_path, delimiter=',')
        kp = torch.tensor(kp) * (512 / img_size)
        kp[kp < 0] = 0
        
        # Generate NTH visualization
        kp_array = np.array(kp, dtype='float32')
        nth = get_nth(kp_array, [512, 512, 3], black=True)
        nth = self.transform(nth)
        save_image(nth, output_dirs['nth'] / face_img_name, normalize=True)
        
        # Generate segmentation masks
        image_seg_input = self.transform_resize(image_raw)
        image_seg_input = image_seg_input.unsqueeze(0).cuda()
        
        with torch.no_grad():
            image_seg_output, _ = get_seg(self.seg_model, image_seg_input, image_seg_input.shape[2:], sigmoid=True)
        
        # Hair mask
        mask_hair = get_seg_mask(image_seg_output, region='hair')[0]
        save_image(mask_hair, output_dirs['mask_hair'] / face_img_name, normalize=True)
        
        # Face mask
        mask_face = get_seg_mask(image_seg_output, region='face')[0]
        save_image(mask_face, output_dirs['mask_face'] / face_img_name, normalize=True)
        
        return face_img_name


class DensePosePipeline:
    """Handles DensePose extraction."""
    
    def __init__(self, hairfusion_dir: str):
        self.hairfusion_dir = Path(hairfusion_dir)
        self.densepose_dir = self.hairfusion_dir / 'detectron2' / 'projects' / 'DensePose'
        self.config_path = self.densepose_dir / 'configs' / 'densepose_rcnn_R_50_FPN_s1x.yaml'
        self.model_url = 'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl'
        
    def extract_densepose(self, input_dir: Path, output_dir: Path):
        """Extract DensePose for all images in input_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check which images need processing
        input_images = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg'))
        images_to_process = []
        for img in input_images:
            output_path = output_dir / img.name
            if not output_path.exists():
                images_to_process.append(img)
        
        if not images_to_process:
            print("  All DensePose images already exist, skipping...")
            return
        
        print(f"  Extracting DensePose for {len(images_to_process)} images...")
        
        # Create temporary directory for images to process
        temp_input_dir = output_dir.parent / 'temp_densepose_input'
        temp_input_dir.mkdir(exist_ok=True)
        for img in images_to_process:
            shutil.copy(img, temp_input_dir / img.name)
        
        # Run DensePose
        cmd = [
            'python', str(self.densepose_dir / 'apply_net.py'),
            'show', str(self.config_path), self.model_url,
            str(temp_input_dir),
            'dp_segm',
            '--output', str(output_dir),
            '-v'
        ]
        
        try:
            # Change to DensePose directory for imports
            original_dir = os.getcwd()
            os.chdir(self.densepose_dir)
            subprocess.run(cmd, check=True, capture_output=True)
            os.chdir(original_dir)
        except subprocess.CalledProcessError as e:
            os.chdir(original_dir)
            print(f"  DensePose extraction failed: {e}")
            # Fallback: create black images as placeholder
            for img in images_to_process:
                output_path = output_dir / img.name
                if not output_path.exists():
                    black_img = np.zeros((512, 512, 3), dtype=np.uint8)
                    cv2.imwrite(str(output_path), black_img)
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_input_dir, ignore_errors=True)


class AgnosticPipeline:
    """Generates agnostic images (hair region removed)."""
    
    def __init__(self, dil_size: int = 50):
        self.dil_size = dil_size
        self.k_face_size_mean = 0.26
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def make_agnostic(self, sample_dirs: dict, img_name: str):
        """Generate agnostic image for a single preprocessed image."""
        agnostic_path = sample_dirs['agnostic'] / img_name
        agnostic_mask_path = sample_dirs['agnostic_mask'] / img_name
        
        # Check if already exists
        if agnostic_path.exists() and agnostic_mask_path.exists():
            return True
        
        # Load image
        image_path = sample_dirs['images'] / img_name
        if not image_path.exists():
            return False
        
        image_raw = Image.open(image_path).convert('RGB')
        img_size = image_raw.size[0]
        image = self.transform(image_raw)
        
        # Load keypoints
        keypoints_path = sample_dirs['keypoints'] / img_name.replace('.png', '.txt')
        if not keypoints_path.exists():
            return False
        
        kp = np.loadtxt(keypoints_path, delimiter=',')
        kp = torch.tensor(kp) * (512 / img_size)
        
        # Calculate face size for adaptive dilation
        x_diff = abs(kp[:, 0].max() - kp[:, 0].min()) / 512
        y_diff = abs(kp[:, 1].max() - kp[:, 1].min()) / 512
        face_size_mean = (x_diff + y_diff) / 2
        
        # Load hair mask with dilation
        hair_path = sample_dirs['mask_hair'] / img_name
        if not hair_path.exists():
            return False
        
        mask_hair = get_binary_from_img(str(hair_path))
        dil_size = int(self.dil_size * (face_size_mean / self.k_face_size_mean))
        dil_size = max(1, dil_size)
        mask_hair_dil = dilation(mask_hair.unsqueeze(0), torch.ones((dil_size, dil_size)))[0]
        
        # Load face mask
        face_path = sample_dirs['mask_face'] / img_name
        if not face_path.exists():
            return False
        
        mask_face = get_binary_from_img(str(face_path))
        
        # Get forehead mask
        mask_forehead = get_forehead(mask_face[0:1].unsqueeze(0), kp.unsqueeze(0))[0]
        mask_forehead_dil = dilation(mask_forehead.unsqueeze(0), torch.ones((5, 5)))[0]
        
        # Face without forehead (to preserve)
        mask_face_wo_fh = mask_face * (1 - mask_forehead_dil)
        mask_face_wo_fh = erosion(mask_face_wo_fh.unsqueeze(0), torch.ones((5, 5)))[0]
        mask_face_wo_fh = dilation(mask_face_wo_fh.unsqueeze(0), torch.ones((3, 3)))[0]
        
        # Load DensePose for boundary calculation
        dp_path = sample_dirs['densepose'] / img_name
        if not dp_path.exists():
            # Try jpg extension
            dp_path = sample_dirs['densepose'] / img_name.replace('.png', '.jpg')
        if not dp_path.exists():
            # Create simple agnostic without DensePose guidance
            agnostic = image.clone()
            agnostic *= (1 - mask_hair_dil)
            agnostic[mask_face_wo_fh > 0] = image[mask_face_wo_fh > 0]
            agnostic_mask = (agnostic != 0) * 1.0
            save_image(agnostic, agnostic_path, normalize=True)
            save_image(agnostic_mask, agnostic_mask_path, normalize=True)
            return True
        
        dp_original = Image.open(dp_path).convert('RGB')
        dp_original = self.transform(dp_original)
        dp_original_1ch = torch.sum(dp_original, axis=0, keepdims=True)
        dp_mask = (dp_original_1ch > 0.2) * 1
        
        # Calculate boundaries using keypoints and DensePose
        dp_nonzero = dp_mask.nonzero()
        if dp_nonzero.numel() > 0:
            start_y = torch.min(dp_nonzero[:, 1])
        else:
            start_y = 0
        
        l_end_x, l_mid_x = kp[0, 0], (kp[6, 0] + kp[7, 0]) / 2
        r_end_x, r_mid_x = kp[16, 0], (kp[9, 0] + kp[10, 0]) / 2
        end_y = max(kp[17:27, 1])
        
        interval = (end_y - start_y) / 5
        start_y = max(0, start_y - interval)
        
        l_wid = 3 * abs(l_end_x - l_mid_x)
        r_wid = 3 * abs(r_end_x - r_mid_x)
        l_start_x = max(0, l_mid_x - l_wid)
        l_end_x = l_mid_x
        r_end_x = min(r_mid_x + r_wid, 512)
        r_start_x = r_mid_x
        
        start_y, end_y = int(start_y), int(end_y)
        l_start_x, l_end_x = int(l_start_x), int(l_end_x)
        r_start_x, r_end_x = int(r_start_x), int(r_end_x)
        
        # Extend bounds based on hair mask
        hair_mask_1ch = torch.sum(mask_hair_dil, axis=0, keepdims=True)
        hair_mask_1ch = (hair_mask_1ch > 0.2) * 1
        hair_nonzero = hair_mask_1ch.nonzero()
        
        if hair_nonzero.numel() > 0:
            start_x_hair = torch.min(hair_nonzero[:, 2])
            end_x_hair = torch.max(hair_nonzero[:, 2])
            l_start_x = min(l_start_x, int(start_x_hair))
            r_end_x = max(r_end_x, int(end_x_hair))
        
        # Create agnostic premask
        agnostic_premask = torch.zeros_like(image)
        agnostic_premask[:, end_y:, l_start_x:l_end_x] = 1
        agnostic_premask[:, end_y:, r_start_x:r_end_x] = 1
        agnostic_premask[:, start_y:end_y, l_start_x:r_end_x] = 1
        
        # Generate agnostic image
        agnostic = image.clone()
        agnostic[agnostic_premask == 1] = 0
        agnostic *= (1 - mask_hair_dil)
        agnostic[mask_face_wo_fh > 0] = image[mask_face_wo_fh > 0]
        
        # Create and clean agnostic mask
        agnostic_mask = (agnostic != 0) * 1.0
        agnostic_mask = dilation(agnostic_mask.unsqueeze(0), torch.ones((10, 10)))[0]
        agnostic_mask = erosion(agnostic_mask.unsqueeze(0), torch.ones((10, 10)))[0]
        agnostic = image * agnostic_mask
        
        # Save
        save_image(agnostic, agnostic_path, normalize=True)
        save_image(agnostic_mask, agnostic_mask_path, normalize=True)
        
        return True


class PairDataset(Dataset):
    """Dataset for a single source-target pair."""
    
    def __init__(self, src_dirs: dict, tgt_dirs: dict, src_img_name: str, tgt_img_name: str, 
                 img_H: int = 512, img_W: int = 512):
        self.src_dirs = src_dirs
        self.tgt_dirs = tgt_dirs
        self.src_img_name = src_img_name
        self.tgt_img_name = tgt_img_name
        self.img_H = img_H
        self.img_W = img_W

    def __len__(self):
        return 1

    @staticmethod
    def imread_for_albu(p, h, w, is_mask=False):
        img = cv2.imread(str(p))
        if not is_mask:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (w, h))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (w, h))
            img = (img >= 128).astype(np.float32)
            img = np.uint8(img * 255.0)
        return img

    @staticmethod
    def norm_for_albu(img, is_mask=False):
        if not is_mask:
            img = (img.astype(np.float32) / 127.5) - 1.0
        else:
            img = img.astype(np.float32) / 255.0
            img = img[:, :, None]
        return img

    def __getitem__(self, idx):
        # Source Identity
        image = self.imread_for_albu(self.src_dirs['images'] / self.src_img_name, self.img_H, self.img_W)
        agn = self.imread_for_albu(self.src_dirs['agnostic'] / self.src_img_name, self.img_H, self.img_W)
        agn_mask = self.imread_for_albu(self.src_dirs['agnostic_mask'] / self.src_img_name, self.img_H, self.img_W, is_mask=True)
        image_nth = self.imread_for_albu(self.src_dirs['nth'] / self.src_img_name, self.img_H, self.img_W)
        
        # Try both extensions for DensePose
        dp_path = self.src_dirs['densepose'] / self.src_img_name
        if not dp_path.exists():
            dp_path = self.src_dirs['densepose'] / self.src_img_name.replace('.png', '.jpg')
        image_keypoints = self.imread_for_albu(dp_path, self.img_H, self.img_W)
        
        hair_mask_src = self.imread_for_albu(self.src_dirs['mask_hair'] / self.src_img_name, self.img_H, self.img_W, is_mask=True)
        
        # Target Hair
        hair = self.imread_for_albu(self.tgt_dirs['images'] / self.tgt_img_name, self.img_H, self.img_W)
        hair_mask = self.imread_for_albu(self.tgt_dirs['mask_hair'] / self.tgt_img_name, self.img_H, self.img_W, is_mask=True)
        hair_nth = self.imread_for_albu(self.tgt_dirs['nth'] / self.tgt_img_name, self.img_H, self.img_W)
        
        tgt_dp_path = self.tgt_dirs['densepose'] / self.tgt_img_name
        if not tgt_dp_path.exists():
            tgt_dp_path = self.tgt_dirs['densepose'] / self.tgt_img_name.replace('.png', '.jpg')
        hair_keypoints = self.imread_for_albu(tgt_dp_path, self.img_H, self.img_W)
        
        # Normalize
        agn = self.norm_for_albu(agn)
        agn_mask = self.norm_for_albu(agn_mask, is_mask=True)
        hair = self.norm_for_albu(hair)
        hair_mask = self.norm_for_albu(hair_mask, is_mask=True)
        image = self.norm_for_albu(image)
        image_keypoints = self.norm_for_albu(image_keypoints)
        image_nth = self.norm_for_albu(image_nth)
        hair_keypoints = self.norm_for_albu(hair_keypoints)
        hair_nth = self.norm_for_albu(hair_nth)
        hair_mask_src = self.norm_for_albu(hair_mask_src, is_mask=True)
        
        ref_image = hair.copy()
        hair = hair * hair_mask
        
        return dict(
            agn=agn,
            agn_mask=agn_mask,
            ref_image=ref_image,
            hair=hair,
            image=image,
            image_keypoints=image_keypoints,
            image_nth=image_nth,
            hair_keypoints=hair_keypoints,
            hair_nth=hair_nth,
            txt="",
            img_fn=self.src_img_name,
            hair_fn=self.tgt_img_name,
            hair_mask=hair_mask,
            hair_mask_src=hair_mask_src,
        )


class HairFusionInference:
    """Main HairFusion inference pipeline."""
    
    def __init__(self, hairfusion_dir: str, config_name: str = "config", 
                 model_path: str = None, vae_path: str = None):
        self.hairfusion_dir = Path(hairfusion_dir)
        
        # Set default paths
        if model_path is None:
            model_path = self.hairfusion_dir / 'logs' / 'hairfusion' / 'models' / '[Train]_[epoch=599]_[train_loss_epoch=0.3666].ckpt'
        if vae_path is None:
            vae_path = self.hairfusion_dir / 'models' / 'realisticVisionV51_v51VAE.ckpt'
            if not vae_path.exists():
                # Try alternative VAE names
                for vae_name in ['realisticVisionV60B1_v51VAE (1).safetensors', 
                                 'realisticVisionV51_v51VAE.ckpt']:
                    alt_path = self.hairfusion_dir / 'models' / vae_name
                    if alt_path.exists():
                        vae_path = alt_path
                        break
        
        self.model_path = Path(model_path)
        self.vae_path = Path(vae_path)
        
        # Load config
        config_path = self.hairfusion_dir / 'configs' / f'{config_name}.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        self.config = self._build_config(config_path)
        self.model = None
        self.ddim_sampler = None
        
    def _build_config(self, config_path):
        config = OmegaConf.load(config_path)
        config.model.params.setdefault("use_VAEdownsample", False)
        config.model.params.setdefault("use_imageCLIP", False)
        config.model.params.setdefault("use_lastzc", False)
        config.model.params.setdefault("use_regdecoder", False)
        config.model.params.setdefault("use_pbe_weight", False)
        config.model.params.setdefault("u_cond_percent", 0.0)
        if not config.model.params.get("validation_config", None):
            config.model.params.validation_config = OmegaConf.create()
        config.model.params.validation_config.ddim_steps = config.model.params.validation_config.get("ddim_steps", 50)
        config.model.params.validation_config.eta = config.model.params.validation_config.get("eta", 0.0)
        config.model.params.validation_config.scale = config.model.params.validation_config.get("scale", 1.0)
        config.model.params.img_H = 512
        config.model.params.img_W = 512
        return config
    
    def load_model(self):
        """Load the HairFusion model."""
        if self.model is not None:
            return
        
        print("Loading HairFusion model...")
        config_path = self.hairfusion_dir / 'configs' / 'config.yaml'
        
        self.model = create_model(config_path=str(config_path), config=self.config)
        self.model.load_state_dict(load_state_dict(str(self.model_path), location="cpu"))
        
        # Load VAE
        if self.vae_path.exists():
            state_dict = load_state_dict(str(self.vae_path), location="cpu")
            new_state_dict = {}
            for k, v in state_dict.items():
                if 'realistic' in str(self.vae_path).lower() or 'safetensors' in str(self.vae_path).lower():
                    if "first_stage_model." in k and "loss." not in k:
                        new_k = k.replace("first_stage_model.", "")
                        new_state_dict[new_k] = v.clone()
                else:
                    if "loss." not in k:
                        new_state_dict[k] = v.clone()
            if new_state_dict:
                self.model.first_stage_model.load_state_dict(new_state_dict)
        
        self.model = self.model.cuda()
        self.model.eval()
        
        self.ddim_sampler = DDIMSampler(
            self.model,
            resampling_trick=False,
            last_n_blend=10,
            resampling_trick_repeat=10
        )
        print("Model loaded.")

    @torch.no_grad()
    def run_inference(self, batch, ddim_steps=50, scale=5.0, eta=0.0):
        """Run inference on a single batch."""
        params = self.config.model.params
        img_H, img_W = 512, 512
        
        z, c = self.model.get_input(batch, params.first_stage_key)
        bs = z.shape[0]
        
        c_crossattn = c["c_crossattn"][0][:bs]
        if c_crossattn.ndim == 4:
            c_crossattn = self.model.get_learned_conditioning(c_crossattn)
            c["c_crossattn"] = [c_crossattn]
        
        mask = batch["agn_mask"]
        x0, _ = self.model.get_input(batch, params.first_stage_key)
        
        if len(mask.shape) == 3:
            mask = mask[..., None]
        mask = rearrange(mask, 'b h w c -> b c h w')
        mask = mask.to(memory_format=torch.contiguous_format).float()
        mask = mask.to(x0.device)
        mask = F.interpolate(mask, (img_H // 8, img_W // 8), mode='nearest')
        
        uc_cross = self.model.get_unconditional_conditioning(bs)
        uc_cat = c["c_concat"]
        uc_full = {"c_concat": uc_cat, "c_crossattn": [uc_cross]}
        uc_full["first_stage_cond"] = c["first_stage_cond"]
        
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        
        self.ddim_sampler.model.batch = batch
        
        shape = (4, img_H // 8, img_W // 8)
        samples, _, _ = self.ddim_sampler.sample(
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
            log_every_t=100
        )
        
        x_samples = self.model.decode_first_stage(samples)
        return x_samples


def get_preprocessed_dirs(base_dir: Path) -> dict:
    """Get dictionary of preprocessing output directories."""
    return {
        'images': base_dir / 'images',
        'keypoints': base_dir / 'keypoints',
        'mask_hair': base_dir / 'mask_hair',
        'mask_face': base_dir / 'mask_face',
        'nth': base_dir / 'nth',
        'densepose': base_dir / 'images-densepose',
        'agnostic': base_dir / 'agnostic',
        'agnostic_mask': base_dir / 'agnostic-mask',
    }


def create_directories(dirs: dict):
    """Create all directories in the dictionary."""
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)


def find_image(image_dir: Path, sample_id: str) -> Path:
    """Find image file with various extensions."""
    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
        path = image_dir / f'{sample_id}{ext}'
        if path.exists():
            return path
    return None


def main(args):
    """Main function for batch processing hair transfer."""
    data_dir = Path(args.data_dir)
    hairfusion_dir = Path(HAIRFUSION_DIR)
    
    # Load pairs from CSV
    pairs_csv_path = data_dir / 'pairs.csv'
    if not pairs_csv_path.exists():
        raise FileNotFoundError(f"pairs.csv not found in {data_dir}")
    
    df = pd.read_csv(pairs_csv_path)
    if 'source_id' not in df.columns or 'target_id' not in df.columns:
        raise ValueError("pairs.csv must contain 'source_id' and 'target_id' columns")
    
    # Shuffle for consistent ordering with other methods
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Setup paths
    image_dir = data_dir / 'image'
    if not image_dir.exists():
        raise FileNotFoundError(f"image folder not found in {data_dir}")
    
    # Output directory
    output_base_dir = data_dir / 'baselines' / 'hairfusion'
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Preprocessing cache directory (shared across all pairs)
    preprocess_cache_dir = data_dir / 'baselines' / 'hairfusion_cache'
    preprocess_dirs = get_preprocessed_dirs(preprocess_cache_dir)
    create_directories(preprocess_dirs)
    
    # Initialize pipelines
    print("Initializing pipelines...")
    preprocess_pipeline = PreprocessingPipeline(str(hairfusion_dir))
    densepose_pipeline = DensePosePipeline(str(hairfusion_dir))
    agnostic_pipeline = AgnosticPipeline(dil_size=args.dil_size)
    inference_pipeline = HairFusionInference(
        str(hairfusion_dir),
        config_name=args.config_name,
        model_path=args.model_path,
        vae_path=args.vae_path
    )
    
    # Collect unique sample IDs
    unique_ids = set(df['source_id'].astype(str).tolist() + df['target_id'].astype(str).tolist())
    print(f"Found {len(unique_ids)} unique sample IDs to preprocess")
    
    # Step 1: Preprocess all unique images
    print("\n=== Step 1: Preprocessing images ===")
    id_to_preprocessed_name = {}
    
    for sample_id in tqdm(unique_ids, desc="Preprocessing"):
        img_path = find_image(image_dir, sample_id)
        if img_path is None:
            print(f"  Warning: Image not found for {sample_id}")
            continue
        
        preprocessed_name = preprocess_pipeline.preprocess_image(
            img_path, preprocess_dirs, crop_scale=args.crop_scale
        )
        if preprocessed_name:
            id_to_preprocessed_name[sample_id] = preprocessed_name
    
    print(f"  Preprocessed {len(id_to_preprocessed_name)} images")
    
    # Step 2: Extract DensePose for all preprocessed images
    print("\n=== Step 2: Extracting DensePose ===")
    densepose_pipeline.extract_densepose(preprocess_dirs['images'], preprocess_dirs['densepose'])
    
    # Step 3: Generate agnostic images (in random order for consistency)
    print("\n=== Step 3: Generating agnostic images ===")
    import random
    agnostic_items = list(id_to_preprocessed_name.items())
    random.seed()
    random.shuffle(agnostic_items)
    for sample_id, img_name in tqdm(agnostic_items, desc="Generating agnostic"):
        agnostic_pipeline.make_agnostic(preprocess_dirs, img_name)
    
    # Step 4: Run inference for each pair (in random order for consistency)
    print("\n=== Step 4: Running hair transfer inference ===")
    inference_pipeline.load_model()
    
    # Shuffle pairs for random processing order
    pair_rows = list(df.iterrows())
    random.seed(42)
    random.shuffle(pair_rows)
    
    for _, row in tqdm(pair_rows, total=len(pair_rows), desc="Hair transfer"):
        source_id = str(row['source_id'])
        target_id = str(row['target_id'])
        
        # Get preprocessed filenames
        src_img_name = id_to_preprocessed_name.get(source_id)
        tgt_img_name = id_to_preprocessed_name.get(target_id)
        
        if src_img_name is None:
            print(f"  Skipping {target_id}_to_{source_id}: source not preprocessed")
            continue
        if tgt_img_name is None:
            print(f"  Skipping {target_id}_to_{source_id}: target not preprocessed")
            continue
        
        # Output paths
        sample_output_dir = output_base_dir / f'{target_id}_to_{source_id}'
        transferred_path = sample_output_dir / 'transferred.png'
        
        # Skip if already exists
        if transferred_path.exists():
            continue
        
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create dataset for this pair
            dataset = PairDataset(
                preprocess_dirs, preprocess_dirs,
                src_img_name, tgt_img_name,
                img_H=512, img_W=512
            )
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            # Run inference
            for batch in dataloader:
                x_samples = inference_pipeline.run_inference(
                    batch,
                    ddim_steps=args.ddim_steps,
                    scale=args.scale,
                    eta=args.eta
                )
                
                # Save output
                x_sample = x_samples[0]
                x_sample_img = tensor2img(x_sample)
                cv2.imwrite(str(transferred_path), x_sample_img[:, :, ::-1])
                
        except Exception as e:
            print(f"  Error processing {target_id}_to_{source_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n=== Processing complete! ===")
    print(f"Results saved to: {output_base_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HairFusion full inference pipeline')
    
    # Data directory
    parser.add_argument('--data_dir', type=str, default='/workspace/outputs',
                        help='Directory containing pairs.csv and aligned_image/ folder')
    
    # Preprocessing parameters
    parser.add_argument('--crop_scale', type=float, default=4.0,
                        help='Scale factor for face cropping')
    parser.add_argument('--dil_size', type=int, default=50,
                        help='Base dilation size for hair mask')
    
    # Model parameters
    parser.add_argument('--config_name', type=str, default='config',
                        help='Name of config file (without .yaml)')
    parser.add_argument('--model_path', type=str, default="/workspace/Baselines/HairFusion/logs/models/[Train]_[epoch=599]_[train_loss_epoch=0.3666].ckpt",
                        help='Path to model checkpoint')
    parser.add_argument('--vae_path', type=str, default="/workspace/Baselines/HairFusion/models/realisticVisionV60B1_v51VAE.safetensors",
                        help='Path to VAE checkpoint')
    
    # Inference parameters
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='Number of DDIM sampling steps')
    parser.add_argument('--scale', type=float, default=5.0,
                        help='Unconditional guidance scale')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM eta parameter')
    
    args = parser.parse_args()
    main(args)
