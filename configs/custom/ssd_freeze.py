_base_ = [
    '../ssdlite_mobilenetv2_scratch_600e_coco.py'
]

dataset_type = 'CocoDataset'

model = dict(
    test_cfg=dict(
        # only output bbox with confidence higher than 0.1
        score_thr=0.1
    ),
    backbone=dict(
        # freeze backbone
        frozen_stages=7,
    ),
)

# batch size 20
data = dict(
    samples_per_gpu=20,
)

# which pretrained model to start with
load_from = 'checkpoints/ssd.pth'

# maximum training epochs
runner = dict(type='EpochBasedRunner', max_epochs=20)
