_base_ = [
    './ssd_base.py'
]

# maximum training epochs
runner = dict(type='EpochBasedRunner', max_epochs=200)

evaluation = dict(
    interval=20,
    metric='bbox',
    classwise=True,
    rule='greater',
    save_best='bbox_mAP_car',
    early_stop=True
)
