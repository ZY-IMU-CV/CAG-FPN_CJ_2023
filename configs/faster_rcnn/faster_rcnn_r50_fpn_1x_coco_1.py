_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    # backbone=dict(
    #     norm_cfg=dict(type='SyncBN', requires_grad=True)),
    neck=dict(
        type='FPN',
        out_channels=512),
    rpn_head=dict(
        in_channels=512,
        # feat_channels=512,
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            out_channels=512),
        bbox_head=dict(
            in_channels=512,)
    )
)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)

data = dict(
    samples_per_gpu=4,
)
work_dir='./work_dirs/AAAAAAAAAAA'