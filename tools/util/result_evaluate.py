from data_split import generate_sample_position

from mmdet.datasets import build_dataset
from mmcv import Config

import math
import pickle
import argparse


def gcd(a, b):
    return a if b == 0 else gcd(b, a % b)


def evaluate_from_file(result, anno_file, img_prefix, downsample=None, config='configs/custom/ssd_amlt.py', **_):
    cfg = Config.fromfile(config)
    cfg.data.test.ann_file = anno_file
    cfg.data.test.img_prefix = img_prefix
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', 1)
    dataset = build_dataset(cfg.data.test)
    if type(result) == str:
        with open(result, 'rb') as f:
            result = pickle.load(f)
    elif type(result) == list:
        result = result
    if downsample:
        n_samples, n_frames = downsample
        c = gcd(n_samples, n_frames)
        n_samples //= c
        n_frames //= c
        positions = generate_sample_position(n_samples, n_frames)
        for start in range(0, len(result), n_frames):
            for j in range(len(positions) - 1):
                for k in range(positions[j] + 1, positions[j + 1]):
                    result[start + k] = result[start + positions[j]]
            for k in range(positions[-1] + 1, n_frames):
                result[start + k] = result[start + positions[-1]]
    return dataset.evaluate(result, metric='bbox', classwise=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMDet evaluate from pickle file')
    parser.add_argument('--result', '-r', help='result file path', type=str, required=True)
    parser.add_argument('--img-prefix', '-p', help='image prefix', type=str, required=True)
    parser.add_argument('--downsample', '-d', help='downsample rate', type=str, default=None)
    parser.add_argument('--anno-file', '-a', help='annotation file path', type=str, required=True)
    parser.add_argument('--config', '-c', help='test config file path', default='configs/custom/ssd.py')
    args = parser.parse_args()
    if args.downsample is not None:
        downsample = args.downsample.split('/')
        args.downsample = (int(downsample[0]), int(downsample[1]))
    evaluation = evaluate_from_file(**args.__dict__)
    # classes_of_interest = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
    classes_of_interest = ['car']
    mAPs_classwise = [evaluation['classwise'][c] for c in classes_of_interest if not math.isnan(evaluation['classwise'][c])]
    print(f'mAP: {evaluation["bbox_mAP"]} classwise: {sum(mAPs_classwise) / len(mAPs_classwise) if len(mAPs_classwise) > 0 else -1:.3f}')
