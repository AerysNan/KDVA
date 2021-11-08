# Copyright (c) OpenMMLab. All rights reserved.
import pickle
import argparse

from mmcv import Config
from mmdet.datasets import build_dataset


def main():
    parser = argparse.ArgumentParser(
        description='MMDet evaluate from pickle file')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('path', help='result file path')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    with open(args.path, 'rb') as f:
        result = pickle.load(f)
    dataset.evaluate(result, metric='bbox')


if __name__ == '__main__':
    main()
