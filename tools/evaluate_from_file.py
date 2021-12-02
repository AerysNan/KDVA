import pickle
import argparse

from mmcv import Config
from mmdet.datasets import build_dataset


def main():
    parser = argparse.ArgumentParser(
        description='MMDet evaluate from pickle file')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('path', help='result file path')
    parser.add_argument(
        '--data', '-d', help='name of dataset', type=str, required=True)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.data.test.ann_file = f'data/annotations/{args.data}_fake.json'
    cfg.data.test.img_prefix = f'data/{args.data}'
    dataset = build_dataset(cfg.data.test)
    with open(args.path, 'rb') as f:
        result = pickle.load(f)
    return dataset.evaluate(result, metric='bbox')


if __name__ == '__main__':
    print(main())
