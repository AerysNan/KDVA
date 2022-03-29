_base_ = [
    './ssd_base.py'
]

runner = dict(type='EpochBasedRunner', max_epochs=40)
