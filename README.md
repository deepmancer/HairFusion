# Hair-Diff


## Notice

This repository only includes the inference codes.


## Installation

Install PyTorch and other dependencies:

```
conda create -y -n [ENV] python=3.8
conda activate [ENV]
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install einops
pip install pytorch-lightning==1.5.0
pip install opencv-python==4.7.0.72
pip install matplotlib
pip install omegaconf
pip install albumentations
pip install transformers==4.33.2
pip install xformers==0.0.19
pip install triton==2.0.0
pip install open-clip-torch==2.19.0
pip install clean-fid==0.1.35
pip install diffusers==0.20.2
pip install scipy==1.10.1
conda install -c anaconda ipython -y
```


## Sample Dataset

sample data in ./data/test_data

## Custom Data
### 1) Preprocessing
Preprocess a given image and save the outputs in ./data/${dir_name}
- Download [face_segment16.pth (50.8MB)](https://drive.google.com/file/d/10GL030sNpVrxM9Ez0nXhHvs9-lsnZFGV/view?usp=sharing), [SGHM-ResNet50.pth (168.1MB)](https://drive.google.com/file/d/1Tl1Nif__Z6tWTyG_PNxA3pJC7f5KLAus/view?usp=sharing), and [shape_predictor_68_face_landmarks.dat (95.1MB)](https://drive.google.com/file/d/1g4jTab8cNVmF2AjDz2N3uXu0cMvsvlC3/view?usp=sharing) into ./models
- run with hair matting:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python preprocess.py --img_path ${image_path} --save_dir_name ${dir_name} --crop_scale ${crop_scale} --hair_matting
  ```
- run without hair matting:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python preprocess.py --img_path ${image_path} --save_dir_name ${dir_name} --crop_scale ${crop_scale}
  ```
- **[Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)**
  - download the 'parsing_atr.onnx' and 'parsing_lip.onnx' in 'humanparsing_checkpoints' from ```mirrorroid-nas``` and save into ./humanparsing/checkpoints
  - add ```--use_humanparsing``` when running ```preprocess.py```

### 2) DensePose Extraction
Extract DensePose of the images in ./data/${dir_name}/images and save them in ./data/${dir_name}/images-densepose
- Installation
    ```bash
    pip install opencv-python torchgeometry Pillow tqdm tensorboardX scikit-image scipy
    cd detectron2
    pip install -e .
    cd projects/DensePose
    pip install -e .
    ```
- run in ./detectron2/projects/DensePose:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python apply_net.py get configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl ../../../data/${dir_name}/images --output ../../../data/${dir_name}/images-densepose  -v
    ```
    Please check the 'detectron2' repository for details.

- DensePose Erosion:
  ```bash
  python dp_erosion.py --dp_dir ./data/${dir_name}/images-densepose
  ```
  The eroded DensePoses will be saved in ./data/${dir_name}/images-densepose-ero

### 3) Make Agnostic Images
- run:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python make_agnostic.py --dir_name ${dir_name} --agn_version ${agnostic_version} --dil_size ${hair_dilation_size}
  ```
  ${agnostic_version} can be 1, 1-3, or 5. 1 only removes hair region, 1-3 is an improved version of 1, and 5 removes part of background regions as well.

### 4) Make Test Pairs
Save a text file of test pairs as ./data/${dir_name}/test_pairs.txt
```bash
python make_test_pairs.py --dir_path ./data/${dir_name}
```

## Checkpoints
- VAE model:
  - download [realisticVisionV51_v51VAE.ckpt (3.97G)](https://drive.google.com/file/d/1eOJUILFwp5MEDX2sPnlpE7MR3_iuFClc/view?usp=sharing) into ./models
- filtered agnostic v8:
  - download a folder named '20240519_filtered_agn8_1dp_2nth_hair5_fg5_face5_control8_aug' (9.5G) from ```mirrorroid-nas``` and save into ./logs
  - download 'Hair_warpingcldm_ca10_cond2.ckpt' (6.85G) from ```mirrorroid-nas``` and save into ./models
- filtered agnostic v8 without DensePose:
  - download a folder named '20240531_filtered_agn8_2nth_aug' (9.5G) from ```mirrorroid-nas``` and save into ./logs
  - download 'Hair_warpingcldm_ca10_cond1.ckpt' (6.85G) from ```mirrorroid-nas``` and save into ./models
- filtered agnostic v8 without DensePose + face drop augmentation:
  - download a folder named '20240604_filtered_agn8_2nth_face_drop3' (9.5G) from ```mirrorroid-nas``` and save into ./logs
  - download 'Hair_warpingcldm_ca10_cond1.ckpt' (6.85G) from ```mirrorroid-nas``` and save into ./models
- filtered agnostic v8 without DensePose + clip hair crop:
  - download a folder named '20240618_filtered_agn8_2nth_aug_clip' (9.5G) from ```mirrorroid-nas``` and save into ./logs
  - download 'Hair_warpingcldm_ca10_cond1.ckpt' (6.85G) from ```mirrorroid-nas``` and save into ./models
- filtered agnostic v8 with align network:
  - download a folder named '20240707_filtered_agn8_1nth_align' (9.5G) from ```mirrorroid-nas``` and save into ./logs
  - download 'Hair_warpingcldm_ca10_cond1_control4_align.ckpt' (6.85G) from ```mirrorroid-nas``` and save into ./models
- filtered agnostic v8 with align + nth + head point:
  - download a folder named '20240715_filtered_agn8_1nth2_align' (9.5G) from ```mirrorroid-nas``` and save into ./logs
  - download 'Hair_warpingcldm_ca10_cond1_control4_align.ckpt' (6.85G) from ```mirrorroid-nas``` and save into ./models
- filtered agnostic v8 with align + nth + hair line:
  - download a folder named '20240715_filtered_agn8_1nth3_align' (9.5G) from ```mirrorroid-nas``` and save into ./logs
  - download 'Hair_warpingcldm_ca10_cond1_control4_align.ckpt' (6.85G) from ```mirrorroid-nas``` and save into ./models
- **filtered agnostic v8 with align + add pose (new)**:
  - download a folder named '20240912_filtered_agn8_0nth_align_noflip_add_pose_dp_bottleneck' (9.5G) from ```mirrorroid-nas``` and save into ./logs
  - download 'Hair_warpingcldm_ca10_cond0_control4_align_add_pose3.ckpt' (6.85G) from ```mirrorroid-nas``` and save into ./models


## Inference

To generate the images using other test dataset, run:
- filtered agnostic v8:
```
CUDA_VISIBLE_DEVICES=0 python inference.py --data_root_dir data/${dir_name}/ --config_name Otherhairstyle_cond2_control8 --save_name other_test_results --agn_name "agnostic8" --agn_mask_name "agnostic-mask8" --model_load_path logs/20240519_filtered_agn8_1dp_2nth_hair5_fg5_face5_control8_aug/models/[Train]_[epoch=499]_[train_loss_epoch=0.3641].ckpt --vae_load_path "models/realisticVisionV51_v51VAE.ckpt" --first_n_repaint ${number of steps to repaint e.g., 10, 20, 30}
```
- filtered agnostic v8 without DensePose:
```
CUDA_VISIBLE_DEVICES=0 python inference.py --data_root_dir data/${dir_name}/ --config_name Otherhairstyle_cond1_control8 --save_name other_test_results --agn_name "agnostic8" --agn_mask_name "agnostic-mask8" --model_load_path logs/20240531_filtered_agn8_2nth_aug/models/[Train]_[epoch=499]_[train_loss_epoch=0.3474].ckpt --vae_load_path "models/realisticVisionV51_v51VAE.ckpt" --first_n_repaint ${number of steps to repaint e.g., 10, 20, 30}
```
- filtered agnostic v8 without DensePose + face drop augmentation:
```
CUDA_VISIBLE_DEVICES=0 python inference.py --data_root_dir data/${dir_name}/ --config_name Otherhairstyle_cond1_control8 --save_name other_test_results --agn_name "agnostic8" --agn_mask_name "agnostic-mask8" --model_load_path logs/20240604_filtered_agn8_2nth_face_drop3/models/[Train]_[epoch=499]_[train_loss_epoch=0.3751].ckpt --vae_load_path "models/realisticVisionV51_v51VAE.ckpt" --first_n_repaint ${number of steps to repaint e.g., 10, 20, 30}
```
- filtered agnostic v8 without DensePose + clip hair crop:
```
CUDA_VISIBLE_DEVICES=0 python inference.py --data_root_dir data/${dir_name}/ --config_name Otherhairstyle_cond1_control8 --save_name other_test_results --agn_name "agnostic8" --agn_mask_name "agnostic-mask8" --model_load_path logs/20240618_filtered_agn8_2nth_aug_clip/models/[Train]_[epoch=499]_[train_loss_epoch=0.3495].ckpt --vae_load_path "models/realisticVisionV51_v51VAE.ckpt" --first_n_repaint ${number of steps to repaint e.g., 10, 20, 30} --clip_hair_crop
```
- filtered agnostic v8 with align network:
```
CUDA_VISIBLE_DEVICES=0 python inference.py --data_root_dir data/${dir_name}/ --config_name Otherhairstyle_cond1_control4_align --save_name other_test_results --agn_name "agnostic8" --agn_mask_name "agnostic-mask8" --model_load_path logs/20240707_filtered_agn8_1nth_align/models/[Train]_[epoch=599]_[train_loss_epoch=0.3666].ckpt --vae_load_path "models/realisticVisionV51_v51VAE.ckpt" --first_n_repaint ${number of steps to repaint e.g., 10, 20, 30}
```
- filtered agnostic v8 with align + nth + head point:
```
CUDA_VISIBLE_DEVICES=0 python inference.py --data_root_dir data/${dir_name}/ --config_name Otherhairstyle_cond1_control4_align --save_name other_test_results --agn_name "agnostic8" --agn_mask_name "agnostic-mask8" --nth_dir 'nth_point' --model_load_path logs/20240715_filtered_agn8_1nth2_align/models/[Train]_[epoch=599]_[train_loss_epoch=0.3683].ckpt --vae_load_path "models/realisticVisionV51_v51VAE.ckpt" --first_n_repaint ${number of steps to repaint e.g., 10, 20, 30}
```
- filtered agnostic v8 with align + nth + hair line:
```
CUDA_VISIBLE_DEVICES=0 python inference.py --data_root_dir data/${dir_name}/ --config_name Otherhairstyle_cond1_control4_align --save_name other_test_results --agn_name "agnostic8" --agn_mask_name "agnostic-mask8" --nth_dir 'nth_line' --model_load_path logs/20240715_filtered_agn8_1nth3_align/models/[Train]_[epoch=599]_[train_loss_epoch=0.3663].ckpt --vae_load_path "models/realisticVisionV51_v51VAE.ckpt" --first_n_repaint ${number of steps to repaint e.g., 10, 20, 30}
```
- **filtered agnostic v8 with align + add pose + repaint using cross attention (CA) (new)**:
```
CUDA_VISIBLE_DEVICES=0 python inference.py --data_root_dir data/${dir_name}/ --config_name Otherhairstyle_cond0_control4_align_add_pose3 --save_name other_test_results --agn_name "agnostic8" --agn_mask_name "agnostic-mask8" --model_load_path logs/20240912_filtered_agn8_0nth_align_noflip_add_pose_dp_bottleneck/models/[Train]_[epoch=599]_[train_loss_epoch=0.3075].ckpt --vae_load_path "models/realisticVisionV51_v51VAE.ckpt" --first_n_repaint ${number of steps to repaint e.g., 10, 20, 30} --last_n_x0_repaint 10 ${number of last steps to repaint using CA e.g., 10, 20, 30}
```

- GPU: TITAN RTX
- Memory: 22.549G per batch
- Step: 50 step per batch
- Inference time: about 3 min per batch
