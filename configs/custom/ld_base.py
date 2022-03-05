_base_ = [
    '../ld_r18_gflv1_r101_fpn_coco_1x.py'
]

# batch size 20
data = dict(
    samples_per_gpu=20,
)

# which pretrained model to start with
load_from = 'checkpoints/ld.pth'

# maximum training epochs
runner = dict(type='EpochBasedRunner', max_epochs=20)

workflow = [('train', 1)]
