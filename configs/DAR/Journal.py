norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
type='force_lss_BEVSegmentor',
pretrained='./configs/_base_/models/swin_tiny_patch4_window7_224.pth',
backbone=dict(
    type='SwinTransformer',
    pretrain_img_size=224,
    embed_dims=96,
    patch_size=4,
    window_size=7,
    mlp_ratio=4,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    strides=(4, 2, 2, 2),
    out_indices=(0, 1, 2, 3),
    qkv_bias=True,
    qk_scale=None,
    patch_norm=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    use_abs_pos_embed=False,
    act_cfg=dict(type='GELU'),
    norm_cfg=dict(type='LN', requires_grad=True),
    pretrain_style='official',
    output_missing_index_as_none=True),
transformer=dict(type='DGKT'),
decode_head=dict(type='Polarhead', num_classes=14, align_corners=True),
train_cfg=dict(),
test_cfg=dict(mode='whole', output_type='iou', positive_thred=0.5))
dataset_type = 'NuscenesDataset'
data_root = './data/nuscenes/'
img_norm_cfg = dict(
mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
dict(type='LoadImageFromFile'),
dict(
    type='LoadAnnotations',
    reduce_zero_label=False,
    with_calib=True,
    imdecode_backend='pyramid'),
dict(
    type='Resize',
    img_scale=(1024, 1024),
    resize_gt=False,
    keep_ratio=False),
dict(type='PhotoMetricDistortion'),
dict(
    type='Normalize',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True),
dict(type='DefaultFormatBundle'),
dict(
    type='Collect',
    keys=['img', 'gt_semantic_seg'],
    meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                'pad_shape', 'scale_factor', 'img_norm_cfg', 'calib'))
]
test_pipeline = [
dict(type='LoadImageFromFile'),
dict(
    type='LoadAnnotations',
    reduce_zero_label=False,
    with_calib=True,
    imdecode_backend='pyramid'),
dict(
    type='MultiScaleFlipAug',
    img_scale=(1024, 1024),
    flip=False,
    transforms=[
        dict(type='Resize', keep_ratio=False, resize_gt=False),
        dict(type='RandomFlip'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='ImageToTensor', keys=['img']),
        dict(
            type='Collect',
            keys=['img'],
            meta_keys=('filename', 'ori_filename', 'ori_shape',
                        'img_shape', 'pad_shape', 'scale_factor', 'flip',
                        'flip_direction', 'img_norm_cfg', 'gt_semantic_seg',
                        'calib'))
    ])
]
data = dict(
samples_per_gpu=6,
workers_per_gpu=2,
train=dict(
    type='NuscenesDataset',
    data_root='./data/nuscenes/',
    img_dir='train/img_dir',
    ann_dir='train/ann_bev_dir',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='LoadAnnotations',
            reduce_zero_label=False,
            with_calib=True,
            imdecode_backend='pyramid'),
        dict(
            type='Resize',
            img_scale=(1024, 1024),
            resize_gt=False,
            keep_ratio=False),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['img', 'gt_semantic_seg'],
            meta_keys=('filename', 'ori_filename', 'ori_shape',
                        'img_shape', 'pad_shape', 'scale_factor',
                        'img_norm_cfg', 'calib'))
    ]),
val=dict(
    type='NuscenesDataset',
    data_root='./data/nuscenes/',
    img_dir='val/img_dir',
    ann_dir='val/ann_bev_dir',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='LoadAnnotations',
            reduce_zero_label=False,
            with_calib=True,
            imdecode_backend='pyramid'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1024, 1024),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=False, resize_gt=False),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(
                    type='Collect',
                    keys=['img'],
                    meta_keys=('filename', 'ori_filename', 'ori_shape',
                                'img_shape', 'pad_shape', 'scale_factor',
                                'flip', 'flip_direction', 'img_norm_cfg',
                                'gt_semantic_seg', 'calib'))
            ])
    ]),
test=dict(
    type='NuscenesDataset',
    data_root='./data/nuscenes/',
    img_dir='val/img_dir',
    ann_dir='val/ann_bev_dir',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='LoadAnnotations',
            reduce_zero_label=False,
            with_calib=True,
            imdecode_backend='pyramid'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1024, 1024),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=False, resize_gt=False),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(
                    type='Collect',
                    keys=['img'],
                    meta_keys=('filename', 'ori_filename', 'ori_shape',
                                'img_shape', 'pad_shape', 'scale_factor',
                                'flip', 'flip_direction', 'img_norm_cfg',
                                'gt_semantic_seg', 'calib'))
            ])
    ]))
log_config = dict(
interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/Xyx/MDT-BEV/IEEEbest_780000.pth'
# load_from = None
resume_from =  '/home/Xyx/MDT-BEV/IEEEbest_780000.pth'
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
type='AdamW',
lr=0.00018,
betas=(0.9, 0.999),
weight_decay=0.01,
paramwise_cfg=dict(
    custom_keys=dict(
        absolute_pos_embed=dict(decay_mult=0.0),
        relative_position_bias_table=dict(decay_mult=0.0),
        norm=dict(decay_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
policy='poly',
warmup='linear',
warmup_iters=1500,
warmup_ratio=1e-06,
power=1.0,
min_lr=0.0,
by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=1000000)
checkpoint_config = dict(by_epoch=False, interval=20000)
evaluation = dict(interval=1000000, metric='mIoU')
backbone_norm_cfg = dict(type='LN', requires_grad=True)
work_dir = './models_dir/QI1KAN1'
gpu_ids = range(0, 1)
