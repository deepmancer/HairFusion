import os
import argparse

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from kornia.morphology import dilation, erosion
from torchvision.utils import save_image
from skimage import io, img_as_float32
from utils import *


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir_name', type=str, default = 'Custom',
                        help='dir name to make agnostic')
    parser.add_argument('--dil_size', type=int, default = 50,
                        help='dilation kernel size')

    return parser.parse_args()

def run():
    args = parse_args()

    data_dir = args.dir_name

    DP_DIR = f'./data/{data_dir}/images-densepose'
    KEYPOINTS_DIR = f'./data/{data_dir}/keypoints'
    IMAGES_DIR = f'./data/{data_dir}/images'
    HAIR_MASK_DIR = f'./data/{data_dir}/mask_hair'
    FACE_MASK_DIR = f'./data/{data_dir}/mask_face'

    AGN_DIR = f'./data/{data_dir}/agnostic'
    AGN_MASK_DIR = f'./data/{data_dir}/agnostic-mask'

    os.makedirs(AGN_DIR, exist_ok=True)
    os.makedirs(AGN_MASK_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img_list = os.listdir(IMAGES_DIR)
    img_list = [f for f in img_list if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(img_list) == 0:
        print(f"No images found in {IMAGES_DIR}")
        print("Please run preprocess.py first to generate cropped images.")
        return
    
    print(f"Found {len(img_list)} images to process")

    k_face_size_mean = 0.26

    for idx, face_img_name in enumerate(img_list, start=1):
        print(f"[{idx}/{len(img_list)}] Processing {face_img_name}...")

        dp_path = os.path.join(DP_DIR, face_img_name)
        # Try both .jpg and .png extensions for DensePose
        dp_path_jpg = os.path.splitext(dp_path)[0] + '.jpg'
        dp_path_png = os.path.splitext(dp_path)[0] + '.png'
        
        if os.path.isfile(dp_path_jpg):
            dp_path = dp_path_jpg
        elif os.path.isfile(dp_path_png):
            dp_path = dp_path_png
        else:
            print(f"  WARNING: No DensePose found for {face_img_name}, skipping...")
            print(f"  (Looked for {dp_path_jpg} and {dp_path_png})")
            continue

        # load image and keypoints
        crop_face_path = os.path.join(IMAGES_DIR, face_img_name)
        image_raw = Image.open(crop_face_path).convert('RGB')
        img_size = image_raw.size[0]
        image = transform(image_raw)

        keypoints_path = os.path.join(KEYPOINTS_DIR, face_img_name.replace('png', 'txt'))
        kp = np.loadtxt(keypoints_path, delimiter=',')  # 68, 2
        kp = torch.tensor(kp) * (512 / img_size)  # 원래 이미지 사이즈 기준으로 뽑아져있음

        x_diff, y_diff = abs(kp[:,0].max() - kp[:,0].min() )/512, abs(kp[:,1].max() - kp[:,1].min() )/512
        face_size_mean = (x_diff+y_diff)/2

        hair_path = os.path.join(HAIR_MASK_DIR, face_img_name)
        mask_hair = get_binary_from_img(hair_path)
        dil_size = int(args.dil_size * (face_size_mean/k_face_size_mean))
        mask_hair_dil = dilation(mask_hair.unsqueeze(0), torch.ones((dil_size, dil_size)))[0]

        face_path = os.path.join(FACE_MASK_DIR, face_img_name)
        mask_face = get_binary_from_img(face_path)

        # make agnostic
        mask_forehead = get_forehead(mask_face[0:1, ].unsqueeze(0), kp.unsqueeze(0))[0]
        mask_forehead_dil = dilation(mask_forehead.unsqueeze(0), torch.ones((5, 5)))[0]

        mask_face_wo_fh = mask_face * (1 - mask_forehead_dil) # 유지할 얼굴 영역 (이마제외)

        # clean mask
        mask_face_wo_fh = erosion(mask_face_wo_fh.unsqueeze(0), torch.ones((5, 5)))[0]
        mask_face_wo_fh = dilation(mask_face_wo_fh.unsqueeze(0), torch.ones((3, 3)))[0]


        dp_original = Image.open(dp_path).convert('RGB')
        dp_original = transform(dp_original)
        dp_original_1ch = torch.sum(dp_original, axis=0, keepdims=True)
        dp_mask = (dp_original_1ch > 0.2) * 1
        start_y = torch.min(dp_mask.nonzero()[:, 1])

        l_end_x, l_mid_x = kp[0, 0], (kp[6, 0] + kp[7, 0]) / 2
        r_end_x, r_mid_x = kp[16, 0], (kp[9, 0] + kp[10, 0]) / 2
        end_y = max(kp[17:26, 1])

        interval = (end_y - start_y) / 5  # 약 15
        start_y = max(0, start_y - interval)
        # print(interval, start_y)
        # start_y = max(0, end_y - abs(end_y - mid_y))
        # start_y = 0
        l_wid = 3 * abs(l_end_x - l_mid_x)
        r_wid = 3 * abs(r_end_x - r_mid_x)
        l_start_x = max(0, l_mid_x - l_wid)
        l_end_x = l_mid_x
        r_end_x = min(r_mid_x + r_wid, 512)
        r_start_x = r_mid_x
        start_y, end_y = int(start_y), int(end_y)
        l_start_x, l_end_x = int(l_start_x), int(l_end_x)
        r_start_x, r_end_x = int(r_start_x), int(r_end_x)

        # hair_mask_original = Image.open(hair_path).convert('RGB')
        # hair_mask_original = transform(hair_mask_original)
        hair_mask_original_1ch = torch.sum(mask_hair_dil, axis=0, keepdims=True)
        hair_mask_1ch = (hair_mask_original_1ch > 0.2) * 1
        
        # Check if hair mask is empty
        hair_nonzero = hair_mask_1ch.nonzero()
        if hair_nonzero.numel() == 0:
            print(f"  WARNING: No hair detected in mask for {face_img_name}, skipping...")
            continue
            
        start_x_hair, end_x_hair = torch.min(hair_nonzero[:, 2]), torch.max(hair_nonzero[:, 2])
        l_start_x = min(l_start_x, start_x_hair)
        r_end_x = max(r_end_x, end_x_hair)

        agnostic_premask = torch.zeros_like(image)
        agnostic_premask[:, end_y:, l_start_x:l_end_x] = 1
        agnostic_premask[:, end_y:, r_start_x:r_end_x] = 1
        agnostic_premask[:, start_y:end_y, l_start_x:r_end_x] = 1

        agnostic = image.clone()
        agnostic[agnostic_premask == 1] = 0

        agnostic *= (1 - mask_hair_dil)
        agnostic[mask_face_wo_fh > 0] = image[mask_face_wo_fh > 0]

        # make agnostic mask
        agnostic_mask = (agnostic != 0) * 1.0
        # clean agnostic mask
        agnostic_mask = dilation(agnostic_mask.unsqueeze(0), torch.ones((10, 10)))[0]
        agnostic_mask = erosion(agnostic_mask.unsqueeze(0), torch.ones((10, 10)))[0]
        agnostic = image * agnostic_mask

        # save agnostic and agnostic mask
        save_image(agnostic, os.path.join(AGN_DIR , face_img_name), normalize=True)
        save_image(agnostic_mask, os.path.join(AGN_MASK_DIR , face_img_name), normalize=True)
    
    print(f"\nDone! Agnostic images saved to {AGN_DIR}")


if __name__ == '__main__':
    run()
