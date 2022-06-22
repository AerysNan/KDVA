_base_ = './rcnn_base.py'
# model settings
data = dict(
    test=dict(
        samples_per_gpu=80,
    )
)
