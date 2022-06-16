_base_ = [
    '../base/ssd_base.py'
]

dataset_type = 'CocoDataset'

model = dict(
    test_cfg=dict(
        score_thr=0.1
    ),
)

# batch size 20
data = dict(
    samples_per_gpu=20,
    test=dict(
        samples_per_gpu=80,
    )
)

# which pretrained model to start with
load_from = 'checkpoints/ssd.pth'

# maximum training epochs
runner = dict(type='EpochBasedRunner', max_epochs=40)

evaluation = dict(interval=200, metric='bbox', classwise=True)
