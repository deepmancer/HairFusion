CUDA_VISIBLE_DEVICES=6 python inference.py --save_name test \
--batch_size 1 --data_root_dir data/test/ --config_name config --last_n_blend 10 \
--model_load_path logs/hairfusion/models/[Train]_[epoch=599]_[train_loss_epoch=0.3666].ckpt