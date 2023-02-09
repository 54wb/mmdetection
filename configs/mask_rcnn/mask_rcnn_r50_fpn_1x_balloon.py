_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

data_root = 'data/balloon/'
classes = ('balloon',)
data = dict(
    samples_per_gpu=4,
    train=dict(
        img_prefix=data_root+'train/',
        classes = classes,
        ann_file = data_root+'annotations/instances_train.json'
    ),
    val=dict(
        img_prefix=data_root+'val/',
        classes = classes,
        ann_file = data_root+'annotations/instances_val.json'
    ),
    test=dict(
        img_prefix=data_root+'val/',
        classes = classes,
        ann_file = data_root+'annotations/instances_val.json',
    )
)
lr_config = dict(
    _delete_=True,
    policy='step',
    step=[8, 11])

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
optimizer = dict(lr=0.005)