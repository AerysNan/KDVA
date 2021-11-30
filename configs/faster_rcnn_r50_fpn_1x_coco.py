_base_ = [
    './faster_rcnn_r50_fpn.py',
    './coco_detection.py',
    './schedule_1x.py',
    './default_runtime.py'
]

dataset_type = 'CocoDataset'
data_root = 'data/'

data = dict(test=dict(
    type=dataset_type,
    ann_file=data_root + 'annotations/sherbrooke.json',
    img_prefix=data_root + 'sherbrooke/'))
