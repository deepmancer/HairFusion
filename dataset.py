import os
from os.path import join as opj
import json
import random
import numpy as np
from glob import glob

import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
import torch
from torchvision.ops import masks_to_boxes
from PIL import Image
import torchvision.transforms as T

import matplotlib.pyplot as plt


DEBUG = False

TRANSFORM_NAMES = ["crop", "hflip", "hsv", "bright_contrast", "shiftscale", "shiftscale2", "shiftscale3","shiftscale4","affine","shiftscale5"]


def get_fn(p):
    return p.split("/")[-1].split(".")[0]


def imread(p, h, w, is_mask=False, in_inverse_mask=False, use_pad=False):
    img = cv2.imread(p)
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not use_pad:
            img = cv2.resize(img, (w, h))
        else:
            img_h, img_w = img.shape[:2]
            ratio = min(h / img_h, w / img_w)
            new_w = int(img_w * ratio)
            new_h = int(img_h * ratio)
            img = cv2.resize(img, (new_w, new_h))
            assert (h - new_h) % 2 == 0 and (w - new_w) % 2 == 0
            margin_h = (h - new_h) // 2
            margin_w = (w - new_w) // 2
            margin_lst = [[margin_h, margin_h], [margin_w, margin_w], [0, 0]]
            img = np.pad(img, margin_lst)

        img = (img.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not use_pad:
            img = cv2.resize(img, (w, h))
        else:
            img_h, img_w = img.shape[:2]
            ratio = min(h / img_h, w / img_w)
            new_w = int(img_w * ratio)
            new_h = int(img_h * ratio)
            img = cv2.resize(img, (new_w, new_h))
            assert (h - new_h) % 2 == 0 and (w - new_w) % 2 == 0
            margin_h = (h - new_h) // 2
            margin_w = (w - new_w) // 2
            margin_lst = [[margin_h, margin_h], [margin_w, margin_w]]
            img = np.pad(img, margin_lst)
        img = (img >= 128).astype(np.float32)  # 0 or 1
        img = img[:, :, None]
        if in_inverse_mask:
            img = 1 - img
    return img


def imread_for_albu(p, h, w, is_mask=False, in_inverse_mask=False, return_size = False):
    img = cv2.imread(p)
    img_h = img.shape[0]
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w, h))
        img = (img >= 128).astype(np.float32)
        if in_inverse_mask:
            img = 1 - img
        img = np.uint8(img * 255.0)
    if return_size:
        return img, img_h
    else:
        return img


def norm_for_albu(img, is_mask=False):
    if not is_mask:
        img = (img.astype(np.float32) / 127.5) - 1.0
    else:
        img = img.astype(np.float32) / 255.0
        img = img[:, :, None]
    return img


def get_nth(img, frame_shape, head=None, hairline=None):
    # lw = 3 #1
    dpi = 100
    fig = plt.figure(figsize=(frame_shape[0] / dpi, frame_shape[1] / dpi), dpi=dpi)

    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.autoscale(tight=True)

    plt.imshow(img)

    # Head point
    if head is not None:
        x, y = head
        ax.plot(x, max(y,3), color='blue', marker='o', markersize=5)
        # ax.scatter(x, y, c='b', marker='o', s=10)

    if hairline is not None:
        y_min, y_max = hairline
        ax.hlines(y_min, xmin=0, xmax=frame_shape[0], colors='blue', linestyles='solid')
        ax.hlines(y_max, xmin=0, xmax=frame_shape[0], colors='aqua', linestyles='solid')

    fig.canvas.draw()

    nth = Image.frombuffer('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(), 'raw', 'RGB', 0, 1)

    plt.close(fig)

    return nth


class CustomDataset(Dataset):
    def __init__(
            self,
            args,
            data_root_dir,
            img_H,
            img_W,
            default_prompt=None,
            is_test=False,
            **kwargs
    ):

        self.args = args
        self.agn_name = "agnostic"
        self.agn_mask_name = "agnostic-mask" 

        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.is_test = is_test

        # Data List - Identity List
        # Test Pair
        self.id1_file_list, self.id2_file_list = [], []
        with open(os.path.join(data_root_dir, 'test_pairs.txt'), 'r') as f:
            for line in f.readlines():
                id1_file, id2_file = line.strip().split()
                self.id1_file_list.append(id1_file)
                self.id2_file_list.append(id2_file)

        print(f"default prompt : {default_prompt}")
        if default_prompt is not None:
            self.prompts = default_prompt
        else:
            self.prompts = ""

        self.nth_dir = "nth"


    def __len__(self):
        return len(self.id1_file_list) if not DEBUG else 4

    def __getitem__(self, idx):
        # Sampling Identity
        src_file = self.id1_file_list[idx]
        tgt_file = self.id2_file_list[idx]

        prompt = self.prompts

        # Source Identity
        image, image_h = imread_for_albu(opj(self.drd, 'images', src_file), self.img_H, self.img_W, return_size=True)
        agn = imread_for_albu(opj(self.drd, self.agn_name, src_file), self.img_H, self.img_W)
        agn_mask = imread_for_albu(opj(self.drd, self.agn_mask_name, src_file), self.img_H, self.img_W, is_mask=True)
        nth_dir = self.nth_dir

        image_nth = imread_for_albu(opj(self.drd, nth_dir, src_file.split('.')[0] + '.png'), self.img_H, self.img_W)
        image_keypoints = imread_for_albu(opj(self.drd, 'images-densepose', src_file.split('.')[0] + '.jpg'), self.img_H, self.img_W)
        hair_mask_src = imread_for_albu(opj(self.drd, 'mask_hair', src_file), self.img_H, self.img_W, is_mask=True)

        # Target Hair
        hair, hair_h = imread_for_albu(opj(self.drd, 'images', tgt_file), self.img_H, self.img_W, return_size=True)
        hair_mask = imread_for_albu(opj(self.drd, 'mask_hair', tgt_file), self.img_H, self.img_W, is_mask=True)
        hair_nth = imread_for_albu(opj(self.drd, self.nth_dir, tgt_file.split('.')[0] + '.png'), self.img_H, self.img_W)
        hair_keypoints = imread_for_albu(opj(self.drd, 'images-densepose', tgt_file.split('.')[0] + '.jpg'), self.img_H, self.img_W)

        # agn_mask = 255 - agn_mask
        agn = norm_for_albu(agn)
        agn_mask = norm_for_albu(agn_mask, is_mask=True)
        hair = norm_for_albu(hair)
        hair_mask = norm_for_albu(hair_mask, is_mask=True)
        image = norm_for_albu(image)
        image_keypoints = norm_for_albu(image_keypoints)
        image_nth = norm_for_albu(image_nth)
        hair_keypoints = norm_for_albu(hair_keypoints)
        hair_nth = norm_for_albu(hair_nth)
        hair_mask_src = norm_for_albu(hair_mask_src, is_mask=True)

        ref_image = hair
        hair = hair * hair_mask


        return dict(
            agn=agn,
            agn_mask=agn_mask,
            ref_image = ref_image,
            hair=hair,
            image=image,
            image_keypoints=image_keypoints,
            image_nth=image_nth,
            hair_keypoints=hair_keypoints,
            hair_nth=hair_nth,
            txt=prompt,
            img_fn=src_file,
            hair_fn=tgt_file,
            hair_mask=hair_mask,
            hair_mask_src=hair_mask_src,

        )