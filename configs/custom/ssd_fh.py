_base_ = [
    './ssd_base.py'
]

model = dict(
    bbox_head=dict(
        frozen=True
    )
)
