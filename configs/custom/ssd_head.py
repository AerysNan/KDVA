_base_ = [
    './ssd_base.py'
]

model = dict(
    backbone=dict(
        # freeze backbone
        frozen_stages=7,
    )
)
