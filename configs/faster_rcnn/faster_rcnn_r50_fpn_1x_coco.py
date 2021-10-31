_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

dataset_type = 'CocoDataset'
data_root = 'data/'

data = dict(test=dict(
    type=dataset_type,
    ann_file=data_root + 'annotations/sherbrooke.json',
    img_prefix=data_root + 'sherbrooke/'))
