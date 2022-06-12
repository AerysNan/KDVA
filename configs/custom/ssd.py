_base_ = [
    '../base/ssdlite_mobilenetv2_scratch_600e_coco.py'
]

dataset_type = 'CocoDataset'

model = dict(
    test_cfg=dict(
        score_thr=0.1
    ),
)

# batch size 60
data = dict(
    samples_per_gpu=20,
    test=dict(
        samples_per_gpu=120,
    )
)

# which pretrained model to start with
load_from = 'checkpoints/ssd.pth'

# maximum training epochs
runner = dict(type='EpochBasedRunner', max_epochs=40)

evaluation = dict(interval=200, metric='bbox', classwise=True)
