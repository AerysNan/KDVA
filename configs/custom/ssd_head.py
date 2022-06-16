_base_ = [
    './ssd.py'
]

model = dict(
    backbone=dict(
        # freeze backbone
        frozen_stages=7,
    )
)
