import json
import argparse

import numpy as np
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from matplotlib.path import Path
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from kornia.morphology import dilation, erosion

from models.face_parsing.model import seg_mean, seg_std

def save_args(args, to_path):
    with open(to_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
def load_args(from_path, is_test=True):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(from_path, "r") as f:
        args.__dict__ = json.load(f)
    args.is_test = is_test
    if "E_name" not in args.__dict__.keys():
        args.E_name = "basic"
    return args   
def tensor2img(x):
    '''
    x : [BS x c x H x W] or [c x H x W]
    '''
    if x.ndim == 3:
        x = x.unsqueeze(0)
    BS, C, H, W = x.shape
    x = x.permute(0,2,3,1).reshape(-1, W, C).detach().cpu().numpy()
    # x = (x+1)/2
    # x = np.clip(x, 0, 1)
    x = np.clip(x, -1, 1)
    x = (x+1)/2
    x = np.uint8(x*255.0)
    if x.shape[-1] == 1:  # gray sclae
        x = np.concatenate([x,x,x], axis=-1)
    return x
def resize_mask(m, shape):
    m = F.interpolate(m, shape)
    m[m > 0.5] = 1
    m[m < 0.5] = 0
    return m

# added
transform_mask = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

def get_binary_from_img(img_path):
    mask_raw = Image.open(img_path).convert('RGB')
    mask_raw = transform_mask(mask_raw)
    binary_mask = (mask_raw < 0.5) * 0.0 + (mask_raw >= 0.5) * 1.0
    return binary_mask


def get_seg(seg_model, input, size, sigmoid=False):
    if input.shape[-1] != 512:
        seg_input = F.interpolate(input.detach().clone(), size=(512,512))
    else:
        seg_input = input.detach().clone()
    seg_input = (seg_input.clamp(0, 1) - seg_mean) / seg_std
    out = seg_model(seg_input)[0]
    if size[0] != 512:
        out = F.interpolate(out, size=size)
    out_seg = torch.argmax(out, dim=1, keepdim=True)
    if sigmoid:
        out_sigmoid = torch.sigmoid(out)
        return out_seg, out_sigmoid
    else:
        return out_seg  # b, 1, 512, 512


def get_crop_coords_crop(keypoints, size, img, scale=2.5):
    min_y, max_y = keypoints[:, 1].min(), keypoints[:, 1].max()
    min_x, max_x = keypoints[:, 0].min(), keypoints[:, 0].max()
    xc = (min_x + max_x) // 2
    # yc = (min_y * 3 + max_y) // 4
    # yc = (min_y + max_y) // 2
    yc = (min_y + max_y * 4) // 5
    h = w = min((max_x - min_x) * scale, min(size[0],size[1]))
    xc = min(max(0, xc - w // 2) + w, size[0]) - w // 2
    yc = min(max(0, yc - h // 2) + h, size[1]) - h // 2
    min_x, max_x = xc - w // 2, xc + w // 2
    min_y, max_y = yc - h // 2, yc + h // 2

    min_y, max_y, min_x, max_x = int(min_y), int(max_y), int(min_x), int(max_x)

    if isinstance(img, np.ndarray):
        return img[min_y:max_y, min_x:max_x]
    else:
        return img.crop((min_x, min_y, max_x, max_y))

# matting
Kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]
def get_unknown_tensor_from_pred(pred, rand_width=30, train_mode=True):
    ### pred: N, 1 ,H, W
    N, C, H, W = pred.shape

    pred = pred.data.cpu().numpy()
    uncertain_area = np.ones_like(pred, dtype=np.uint8)
    uncertain_area[pred<1.0/255.0] = 0
    uncertain_area[pred>1-1.0/255.0] = 0

    for n in range(N):
        uncertain_area_ = uncertain_area[n,0,:,:] # H, W
        if train_mode:
            width = np.random.randint(1, rand_width)
        else:
            width = rand_width // 2
        uncertain_area_ = cv2.dilate(uncertain_area_, Kernels[width])
        uncertain_area[n,0,:,:] = uncertain_area_

    weight = np.zeros_like(uncertain_area)
    weight[uncertain_area == 1] = 1
    weight = torch.from_numpy(weight).cuda()

    return weight

def get_forehead(face_mask_batch, kp_batch):
    # face_mask  B,1,256,256 ( 1,3,256,256)
    # kp B, 68, 2

    output_list = []

    for idx in range(face_mask_batch.shape[0]):
        face_mask = face_mask_batch[idx:idx+1].cpu().numpy() #1, 1, 256, 256
        kp = kp_batch[idx].cpu().numpy()

        contour = np.concatenate((kp[0:17], kp[17:27][::-1]), axis=0)
        path = Path(contour)
        path.get_extents()
        # contains_points
        x, y = np.mgrid[:face_mask.shape[2], :face_mask.shape[3]]
        points = np.vstack((x.ravel(), y.ravel())).T

        landmark_mask = path.contains_points(points)
        # path_points = points[np.where(landmark_mask)]

        img_mask = landmark_mask.reshape(x.shape).T  # 256,256 > boolean
        img_mask_in = (img_mask != True)

        result = face_mask * img_mask_in  # 1,1,256,256
        result = result.squeeze(axis=0)  # 1,256,256
        result = result.transpose(1, 2, 0)  # 256,256,1

        tmp = Image.fromarray(np.array(result.repeat(3, 2)).astype(np.uint8)) # 256,256,3
        kernel = np.ones((3, 3), np.uint8)
        eroded_result = cv2.erode(np.float32(tmp), kernel, iterations=1).astype(np.uint8)  # 256, 256, 3
        # k = Image.fromarray(eroded_result)
        output_list.append(torch.tensor(eroded_result, dtype=torch.float32).permute(2, 0, 1)) # 3, 256, 256

    return torch.stack(output_list)  # return B, 3, 256, 256



def get_clean_mask(input, kernel_size=11):
    assert kernel_size % 2 == 1
    kernel = torch.ones((kernel_size, kernel_size)).cuda()
    input_clean = dilation(erosion(input, kernel), kernel)
    input_clean = erosion(dilation(input_clean, kernel), kernel)
    return input_clean



seg_dict = {'bg': 0, 'skin': 1, 'nose': 2, 'eye_g': 3, 'eyes': 4, 'eyebrows': 5, 'ears': 6, 'mouth': 7, 'u_lip': 8,
            'l_lip': 9, 'hair': 10, 'hat': 11, 'ear_r': 12, 'neck_l': 13, 'neck': 14, 'cloth': 15}


def get_seg_mask(input, region='', sigmoid_input=None):
    # label_list = [bg, 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip',
    #               'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
    # correspond_list = [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8,
    #                       9, 10, 11, 12, 13, 14, 15]
    # face: 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g',  'nose', 'mouth', 'u_lip', 'l_lip' 1, 2, 3, 4, 5,  6, 7, 8, 9
    if region == 'face':
        output = (input >= 1) * (input < 6) * 1.0 + (input >= 6) * (input < 10) * 1.0
        output = (output > 0) * 1.0
    else:
        output = (input == seg_dict[region]) * 1.0
        if region == 'neck':
            output += (input == seg_dict['neck_l']) * 1.0
            output = (output > 0) * 1.0
        if region == 'bg':
            output += (input == seg_dict['ear_r']) * 1.0
            output = (output > 0) * 1.0


    if region != 'neck':
        output = get_clean_mask(output)

    if sigmoid_input is not None:
        sigmoid_output = sigmoid_input[:, seg_dict[region], :, :].unsqueeze(1)
        sigmoid_output *= output
        return output, sigmoid_output
    return output


def get_nth(keypoint, frame_shape, lw=None, black=False, img=None, head=None, hairline=None):
    lw = lw if lw is not None else 1
    dpi = 100
    fig = plt.figure(figsize=(frame_shape[0] / dpi, frame_shape[1] / dpi), dpi=dpi)

    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    plt.autoscale(tight=True)

    if img is not None:
        plt.imshow(img)
    else:
        if black:
            plt.imshow(np.zeros(frame_shape))
        else:
            plt.imshow(np.ones(frame_shape))

    # Head
    ax.plot(keypoint[0:17, 0], keypoint[0:17, 1], linestyle='-', color='green', lw=lw)

    # Eyebrows - for closed shape
    ax.plot(keypoint[17:22, 0], keypoint[17:22, 1], linestyle='-', color='orange', lw=lw)
    ax.plot(keypoint[22:27, 0], keypoint[22:27, 1], linestyle='-', color='orange', lw=lw)

    # Nose
    ax.plot(keypoint[27:31, 0], keypoint[27:31, 1], linestyle='-', color='red', lw=lw)
    ax.plot(keypoint[31:36, 0], keypoint[31:36, 1], linestyle='-', color='red', lw=lw)

    # Eyes - for closed shape
    eyes = np.append(keypoint[36:42], keypoint[36].reshape(1, 2), axis=0)
    ax.plot(eyes[:, 0], eyes[:, 1], linestyle='-', color='purple', lw=lw)

    eyes = np.append(keypoint[42:48], keypoint[42].reshape(1, 2), axis=0)
    ax.plot(eyes[:, 0], eyes[:, 1], linestyle='-', color='brown', lw=lw)

    # Mouth - for closed shape
    mouth = np.append(keypoint[49:60], keypoint[49].reshape(1, 2), axis=0)
    ax.plot(mouth[:, 0], mouth[:, 1], linestyle='-', color='magenta', lw=lw)
    mouth = np.append(keypoint[60:68], keypoint[60].reshape(1, 2), axis=0)
    ax.plot(mouth[:, 0], mouth[:, 1], linestyle='-', color='magenta', lw=lw)

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