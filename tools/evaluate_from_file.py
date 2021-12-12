import pickle
import argparse

from mmcv import Config
from mmdet.datasets import build_dataset


def evaluate_from_file(config, path, data):
    cfg = Config.fromfile(config)
    cfg.data.test.ann_file = f'data/annotations/{data}.gt.json'
    cfg.data.test.img_prefix = ''
    dataset = build_dataset(cfg.data.test)
    with open(path, 'rb') as f:
        result = pickle.load(f)
    return dataset.evaluate(result, metric='bbox')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMDet evaluate from pickle file')
    parser.add_argument('--config', '-c', help='test config file path', default='configs/custom/ssd.py')
    parser.add_argument('--path', '-p', help='result file path', required=True)
    parser.add_argument('--data', '-d', help='name of dataset', type=str, required=True)
    args = parser.parse_args()
    print(evaluate_from_file(args.config, args.path, args.data)['bbox_mAP'])
