#!/usr/bin/python

import os
import sys
import copy
import pickle
import argparse

from mmcv import Config
from mmdet.datasets import build_dataset


parser = argparse.ArgumentParser(
    description='Evaluate system performance')
parser.add_argument('--id', '-i', help='ID of video source to be evaluated', required=True)
parser.add_argument('--dataset', '-d', help='name of dataset', required=True)
parser.add_argument('--config', '-c', help='config file path', default="/home/ubuntu/urban/configs/custom/ssd.py")
args = parser.parse_args()
cfg = Config.fromfile(args.config)
cfg.data.test.ann_file = f'data/annotations/{args.dataset}.json'
cfg.data.test.img_prefix = f'data/{args.dataset}'
dataset = build_dataset(cfg.data.test)

if not os.path.exists(f'dump/result/{args.id}/0.pkl'):
    print('Result files must contain at least the first one')
    sys.exit(1)

result = []
previous = None

for i in range(len(dataset)):
    if os.path.exists(f'dump/result/{args.id}/{i}.pkl'):
        with open(f'dump/result/{args.id}/{i}.pkl', 'rb') as f:
            o = pickle.load(f)
            result.append(o)
            previous = o
    else:
        result.append(copy.deepcopy(previous))


evaluation = dataset.evaluate(result, metric='bbox')
print(f"mAP = {evaluation['bbox_mAP']}")
