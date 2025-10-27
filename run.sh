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


#? 使用small model进行推理/评估
CUDA_VISIBLE_DEVICES=5 python -m src.main \
    +experiment=re10k \
    mode=test \
    model.encoder.upsample_factor=4 \
    model.encoder.lowest_feature_resolution=4 \
    checkpointing.pretrained_model=/home/lianghao/wangyushen/data/wangyushen/Weights/depthsplat/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
    dataset.roots=[/home/lianghao/wangyushen/data/wangyushen/Datasets/re10k/re10k_subset] \
    dataset.test_chunk_interval=1 \
    dataset/view_sampler=evaluation \
    test.output_path=/home/lianghao/wangyushen/data/wangyushen/Output/depth_splat/test \
    test.save_image=true \
    test.save_gt_image=true \
    output_dir=/home/lianghao/wangyushen/data/wangyushen/Output/depth_splat/test

