_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
model = dict(
    backbone=dict(
        depth=50,),
    neck=dict(
        out_channels=512,),
    bbox_head=dict(
        in_channels=512,
        # feat_channels=512,
    )
)
data = dict(
    samples_per_gpu=4,
)
work_dir='./work_dirs/Da_retinanet'