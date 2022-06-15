_base_ = './rcnn.py'
# model settings
data = dict(
    test=dict(
        samples_per_gpu=120,
    )
)
