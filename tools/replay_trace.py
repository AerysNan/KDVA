import copy
import json
import pickle
import argparse

from mmcv import Config
from mmdet.datasets import build_dataset

ORIGINAL_FRAMERATE = 60


def replay_trace(path, name, framerate, batch_size):
    cfg = Config.fromfile('configs/custom/ssd.py')
    with open('datasets.json') as f:
        datasets = json.load(f)
    if not name in datasets:
        key = name[:name.rfind('_')]
    else:
        key = name
    n_epoch = datasets[key]['size'] // batch_size
    if type(framerate) == int:
        framerate = [framerate for _ in range(n_epoch)]
    results = []
    with open(f'{path}/000000.pkl', 'rb') as f:
        previous = pickle.load(f)
    mAP = []
    for i in range(n_epoch):
        begin, end, stride = i * batch_size, i * batch_size + batch_size, ORIGINAL_FRAMERATE // framerate[i]
        cfg.data.test.ann_file = f'data/annotations/{name}_test_{i}.gt.json'
        cfg.data.test.img_prefix = ''
        dataset = build_dataset(cfg.data.test)
        result = []
        for j in range(begin, end):
            if j % stride == 0:
                with open(f'{path}/{j:06d}.pkl', 'rb') as f:
                    o = pickle.load(f)
                    result.append(o)
                    previous = o
            else:
                result.append(copy.deepcopy(previous))
        mAP.append(dataset.evaluate(result, metric='bbox')['bbox_mAP'])
        results.extend(result)
    cfg.data.test.ann_file = f'data/annotations/{key}.gt.json'
    cfg.data.test.img_prefix = ''
    dataset = build_dataset(cfg.data.test)
    mAP.append(dataset.evaluate(results, metric='bbox')['bbox_mAP'])
    return mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Replay a single trace')
    parser.add_argument('--path', '-p', help='path to result files', type=str, required=True)
    parser.add_argument('--dataset', '-d', type=str, required=True, help='name of dataset')
    parser.add_argument('--framerate', '-f', type=int, default=60, help='replay framerate')
    parser.add_argument('--size', '-s', type=int, required=True, help='chunk size')
    args = parser.parse_args()
    mAPs = replay_trace(args.path, args.dataset, args.framerate, args.size)
    for mAP in mAPs:
        print(mAP)
