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
        ann_file = data_root+'annotations/instances_val.json'
    )
)

optimizer = dict(lr=0.005)