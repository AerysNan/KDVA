import json
import argparse


parser = argparse.ArgumentParser(
    description='Generate config file')
parser.add_argument(
    '--path', '-p', help='path to dataset file', type=str, default='datasets.json')
parser.add_argument(
    '--size', '-s', help='size of splitted dataset', type=int, default=600)
parser.add_argument(
    '--epoch', '-e', help='number of epochs to train', type=int, default=20)
parser.add_argument(
    '--postfix', '-o', help='generated postfix', type=str)
args = parser.parse_args()

if args.postfix:
    postfix = f'_{args.postfix}'
else:
    postfix = ''

with open(args.path) as f:
    dataset = json.load(f)

for prefix in dataset:
    n = dataset[prefix]['size'] // args.size

    for i in range(n):
        f = open(f'configs/custom/ssd_{prefix}{postfix}_{i}.py', 'w')
        f.write(f"""
_base_ = '../ssdlite_mobilenetv2_scratch_600e_coco.py'
# Modify dataset related settings
dataset_type = 'CocoDataset'
data_root = 'data/'

model = dict(
    backbone=dict(
        frozen_stages=8,
    )
)

data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/{prefix}{postfix}_train_{i-1}.gt.json',
        img_prefix=''),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/{prefix}{postfix}_val_{i-1}.gt.json',
        img_prefix=''),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/{prefix}{postfix}_test_{i}.gt.json',
        img_prefix='')
)

load_from = 'checkpoints/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth'
runner = dict(type='EpochBasedRunner', max_epochs={args.epoch})
      """)
        f.close()
