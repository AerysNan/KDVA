_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet18')))

dataset_type = 'CocoDataset'
data_root = 'data/'

data = dict(test=dict(
    type=dataset_type,
    ann_file=data_root + 'annotations/sherbrooke.json',
    img_prefix=data_root + 'sherbrooke/'))
