_base_ = '../base/faster_rcnn_r101_fpn_1x_coco.py'
# model settings

model = dict(
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.1
        )
    )
)

data = dict(
    test=dict(
        samples_per_gpu=50,
    )
)
