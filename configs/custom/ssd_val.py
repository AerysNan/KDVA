_base_ = [
    './ssd_base.py'
]

evaluation = dict(interval=1, metric='bbox', classwise=True)
runner = dict(type='EpochBasedRunner', max_epochs=20)
