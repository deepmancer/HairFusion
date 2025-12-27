import os
import argparse

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from kornia.morphology import dilation, erosion
from torchvision.utils import save_image
from skimage import io, img_as_float32

from models.ffhq_dataset.landmarks_detector import LandmarksDetector
from models.face_parsing.model import BiSeNet
from utils import *
from torchvision.ops import masks_to_boxes


def get_crop_coords(keypoints, size, scale=2.5):
    """Get crop coordinates based on keypoints. Returns (min_x, min_y, max_x, max_y)."""
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

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', type=str, required=True,
                        help='directory containing images to preprocess')
    parser.add_argument('--save_dir_name', type=str, default = 'Custom',
                        help='dir name to save preprocessed data')
    parser.add_argument('--crop_scale', type=float, default = 4,
                        help='scale to crop, the bigger, the more zoom out')

    return parser.parse_args()

def run():
    args = parse_args()

    data_dir = args.save_dir_name

    # If data_dir already starts with 'data/', don't add it again
    if data_dir.startswith('data/') or data_dir.startswith('./data/'):
        base_path = f'./{data_dir}' if not data_dir.startswith('./') else data_dir
    else:
        base_path = f'./data/{data_dir}'
    
    # Remove trailing slash if present
    base_path = base_path.rstrip('/')
    
    # Input directory (where original images are)
    img_dir = args.img_dir
    
    # Output directories
    KEYPOINTS_DIR = f'{base_path}/keypoints'
    IMAGES_DIR = f'{base_path}/images'
    HAIR_MASK_DIR = f'{base_path}/mask_hair'
    FACE_MASK_DIR = f'{base_path}/mask_face'
    NTH_DIR = f'{base_path}/nth'
    
    # Check if input and output image directories are the same
    input_abs = os.path.abspath(img_dir)
    output_abs = os.path.abspath(IMAGES_DIR)
    if input_abs == output_abs:
        print("ERROR: Input directory and output images directory are the same!")
        print(f"  Input:  {input_abs}")
        print(f"  Output: {output_abs}")
        print("Please use a different input directory (e.g., 'data/test/raw_images')")
        print("or a different save_dir_name.")
        return

    os.makedirs(KEYPOINTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(HAIR_MASK_DIR, exist_ok=True)
    os.makedirs(FACE_MASK_DIR, exist_ok=True)
    os.makedirs(NTH_DIR, exist_ok=True)


    # keypoint extractor
    landmarks_model_path = './models/shape_predictor_68_face_landmarks.dat'
    if not os.path.isfile(landmarks_model_path):
        print(f'{landmarks_model_path} does not exist!')
        exit()
    landmarks_detector = LandmarksDetector(landmarks_model_path)

    seg_model = BiSeNet(n_classes=16)
    seg_model.cuda()
    seg_model_path = './models/face_segment16.pth'
    if not os.path.isfile(landmarks_model_path):
        print(f'{seg_model_path} does not exist!')
        exit()
    seg_model.load_state_dict(torch.load(seg_model_path))
    seg_model.eval()


    # Get list of image files from input directory
    img_dir = args.img_dir
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(supported_extensions)]
    image_files.sort()
    
    if len(image_files) == 0:
        print(f"No image files found in {img_dir}")
        return
    
    print(f"Found {len(image_files)} image(s) to process")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_resize = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    # Process each image
    for img_idx, img_file in enumerate(image_files, start=1):
        raw_img_path = os.path.join(img_dir, img_file)
        print(f"\n[{img_idx}/{len(image_files)}] Processing {img_file}...")
        
        # Detect faces
        detected_landmarks = landmarks_detector.get_landmarks(raw_img_path)
        if len(detected_landmarks) == 0:
            print(f"  WARNING: No faces detected in {img_file}, skipping...")
            continue
        
        print(f"  Detected {len(detected_landmarks)} face(s)")

        for i, face_landmarks in enumerate(detected_landmarks, start=1):
            face_img_name = '%s_%02d.png' % (os.path.splitext(os.path.basename(raw_img_path))[0], i)
            # save image cropped based on keypoints
            crop_face_path = os.path.join(IMAGES_DIR, face_img_name)
            raw_img = img_as_float32(io.imread(raw_img_path))
            
            # Get crop coordinates and crop the image
            face_landmarks_np = np.array(face_landmarks)
            min_x, min_y, max_x, max_y = get_crop_coords(face_landmarks_np, (raw_img.shape[1], raw_img.shape[0]), scale=args.crop_scale)
            raw_img_cropped = raw_img[min_y:max_y, min_x:max_x]
            crop_w, crop_h = max_x - min_x, max_y - min_y

            raw_img_cropped = (raw_img_cropped * 255).astype(np.uint8)
            raw_img_cropped_pil = Image.fromarray(raw_img_cropped)
            raw_img_cropped_pil.save(crop_face_path)

            # Transform original landmarks to cropped image coordinates
            face_landmarks_cropped = face_landmarks_np.copy()
            face_landmarks_cropped[:, 0] = face_landmarks_cropped[:, 0] - min_x  # adjust x
            face_landmarks_cropped[:, 1] = face_landmarks_cropped[:, 1] - min_y  # adjust y
            
            # save keypoints
            keypoints_path = os.path.join(KEYPOINTS_DIR, face_img_name.replace('png', 'txt'))
            np.savetxt(keypoints_path, face_landmarks_cropped, fmt='%d', delimiter=',')

            # load image and keypoints
            image_raw = Image.open(crop_face_path).convert('RGB')
            img_size = image_raw.size[0]

            kp = np.loadtxt(keypoints_path, delimiter=',')  # 68, 2
            kp = torch.tensor(kp) * (512 / img_size)  # img_size -> 512
            kp[kp<0] = 0

            kp_array = np.array(kp, dtype='float32')
            nth = get_nth(kp_array, [512, 512, 3], black=True)
            nth = transform(nth)
            save_image(nth, os.path.join(NTH_DIR, face_img_name), normalize=True)


            # save hair mask
            image_seg_input = transform_resize(image_raw)
            image_seg_input = image_seg_input.unsqueeze(0).cuda()  # [bs, 3, 256, 256]


            image_seg_output, image_seg_sigmoid = get_seg(seg_model,image_seg_input, image_seg_input.shape[2:], sigmoid=True)

            mask_hair = get_seg_mask(image_seg_output, region='hair')[0]

            hair_path = os.path.join(HAIR_MASK_DIR, face_img_name)
            save_image(mask_hair, hair_path, normalize=True)

            mask_face = get_seg_mask(image_seg_output, region='face')[0]
            face_path = os.path.join(FACE_MASK_DIR, face_img_name)
            save_image(mask_face, face_path, normalize=True)
    
    print(f"\nPreprocessing complete! Output saved to {base_path}")


if __name__ == '__main__':
    run()
