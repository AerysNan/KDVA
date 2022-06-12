_base_ = './ssd.py'
# model settings
data = dict(
    samples_per_gpu=60,
    test=dict(
        samples_per_gpu=360,
    )
)
