_base_ = './ssd_base.py'
# model settings
data = dict(
    test=dict(
        samples_per_gpu=240,
    )
)
