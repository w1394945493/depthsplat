# todo DepthSplat re10k evaluate
# todo base model 参考：
# Table 1 of depthsplat paper
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth \
mode=test \
dataset/view_sampler=evaluation
# todo small model 参考
# Table 1 of depthsplat paper
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
mode=test \
dataset/view_sampler=evaluation


#? 使用small model进行推理/评估 re10k-subset
CUDA_VISIBLE_DEVICES=5 python -m src.main \
    +experiment=re10k \
    mode=demo \
    model.encoder.upsample_factor=4 \
    model.encoder.lowest_feature_resolution=4 \
    checkpointing.pretrained_model=/home/lianghao/wangyushen/data/wangyushen/Weights/depthsplat/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
    dataset.roots=[/home/lianghao/wangyushen/data/wangyushen/Datasets/re10k/re10k_subset] \
    dataset.test_chunk_interval=1 \
    dataset/view_sampler=evaluation \
    test.output_path=/home/lianghao/wangyushen/data/wangyushen/Output/depth_splat/test \
    test.save_image=true \
    test.save_gt_image=true \
    test.save_gaussian=true \
    output_dir=/home/lianghao/wangyushen/data/wangyushen/Output/depth_splat/test


#? 使用small model进行推理/评估 omni-scene demo
CUDA_VISIBLE_DEVICES=5 python -m src.main \
    +experiment=omniscene_224x400 \
    mode=test \
    model.encoder.upsample_factor=4 \
    model.encoder.lowest_feature_resolution=4 \
    checkpointing.pretrained_model=/home/lianghao/wangyushen/data/wangyushen/Weights/depthsplat/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
    dataset.roots=[/home/lianghao/wangyushen/data/wangyushen/Datasets/dataset_omniscene] \
    test.save_image=true \
    test.save_gt_image=true \
    output_dir=/home/lianghao/wangyushen/data/wangyushen/Output/depth_splat/test/omniscene_demo_debug_flip_yz

#? 使用small model进行训练 omni-scene
CUDA_VISIBLE_DEVICES=6 python -m src.main \
    +experiment=omniscene_112x200 \
    mode=train \
    model.encoder.upsample_factor=4 \
    model.encoder.lowest_feature_resolution=4 \
    checkpointing.pretrained_model=/home/lianghao/wangyushen/data/wangyushen/Weights/depthsplat/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
    dataset.roots=[/home/lianghao/wangyushen/data/wangyushen/Datasets/dataset_omniscene] \
    trainer.max_steps=100_001 \
    trainer.val_check_interval=5_000 \
    checkpointing.every_n_train_steps=5_000 \
    train.print_log_every_n_steps=100 \
    output_dir=/home/lianghao/wangyushen/data/wangyushen/Output/depth_splat/train/train_112x200