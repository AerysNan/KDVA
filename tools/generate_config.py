import json
import argparse


parser = argparse.ArgumentParser(
    description='Generate config file')
parser.add_argument(
    '--path', '-p', help='path to dataset file', type=str, default='data/dataset.json')
parser.add_argument(
    '--size', '-s', help='size of splitted dataset', type=int, default=500)
parser.add_argument(
    '--epoch', '-e', help='size of splitted dataset', type=int, default=20)
parser.add_argument(
    '--config', '-c', help='postfix of splitted dataset', type=str, required=True)
args = parser.parse_args()

with open(args.path) as f:
    dataset = json.load(f)

for prefix in dataset:
    n = dataset[prefix] // args.size

    for i in range(n):
        f = open(f'configs/custom/ssd_{prefix}_{i}_{args.config}.py', 'w')
        f.write(f"""
_base_ = '../ssdlite_mobilenetv2_scratch_600e_coco.py'
# Modify dataset related settings
dataset_type = 'CocoDataset'
data_root = 'data/'

data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/{prefix}_train_{i-1}_{args.config}.json',
        img_prefix=data_root + '{prefix}_train_{i-1}_{args.config}/'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/{prefix}_val_{i-1}_{args.config}.json',
        img_prefix=data_root + '{prefix}_val_{i-1}_{args.config}/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/{prefix}_test_{i}_{args.config}.json',
        img_prefix=data_root + '{prefix}_test_{i}_{args.config}/')
)

load_from = 'tmp_{prefix}_{args.config}/latest.pth'
runner = dict(type='EpochBasedRunner', max_epochs={args.epoch})
      """)
        f.close()
