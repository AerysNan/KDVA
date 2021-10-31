import os

prefix = 'virat_1'
n = 20

for i in range(n):
    f = open(f'ssd_{prefix}_{i}.py', 'w')
    f.write(f"""
_base_ = '../ssd/ssdlite_mobilenetv2_scratch_600e_coco.py'
# Modify dataset related settings
dataset_type = 'CocoDataset'
data_root = 'data/'

data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/{prefix}_train_{i-1}.json',
        img_prefix=data_root + '{prefix}_train_{i-1}/'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/virat_1.json',
        img_prefix=data_root + 'virat_1/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/{prefix}_test_{i}.json',
        img_prefix=data_root + '{prefix}_test_{i}/')
)

load_from = 'tmp_{prefix}/previous.pth'
runner = dict(type='EpochBasedRunner', max_epochs=30)
    """)
    f.close()
