#!/usr/bin/python

import os
import copy
import pickle
import argparse

from mmcv import Config
from mmdet.datasets import build_dataset


def evalutate_system(path, ds, config, l=0, r=None):
    cfg = Config.fromfile(config)
    cfg.data.test.ann_file = f'data/annotations/{ds}.gt.json'
    cfg.data.test.img_prefix = ''
    dataset = build_dataset(cfg.data.test)

    result = []
    previous = None

    if not os.path.exists(f'{path}/{l:06d}.pkl'):
        for i in range(l, -1, -1):
            if os.path.exists(f'{path}/{i:06d}.pkl'):
                with open(f'{path}/{i:06d}.pkl', 'rb') as f:
                    previous = pickle.load(f)
                break

    if r is None:
        r = len(dataset)

    for i in range(l, r):
        if os.path.exists(f'{path}/{i:06d}.pkl'):
            with open(f'{path}/{i:06d}.pkl', 'rb') as f:
                o = pickle.load(f)
                result.append(o)
                previous = o
        else:
            result.append(copy.deepcopy(previous))

    evaluation = dataset.evaluate(result, metric='bbox')
    return evaluation['bbox_mAP']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate system performance')
    parser.add_argument('--path', '-p', help='path to result files', type=str, required=True)
    parser.add_argument('--dataset', '-d', help='name of dataset', type=str, required=True)
    parser.add_argument('--config', '-c', help='config file path', type=str, default="/home/ubuntu/urban/configs/custom/ssd.py")
    args = parser.parse_args()
    print(f'mAP = {evalutate_system(args.path, args.dataset, args.config)}')
