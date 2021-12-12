import sys
import copy
import json
import numpy as np
import pickle
import argparse

from mmcv import Config
from mmdet.datasets import build_dataset

ORIGINAL_FRAMERATE = 60


def allocate(bottleneck, profit_matrix):
    n_config, n_stream = profit_matrix.shape
    # start DP
    dp_record = np.full((n_stream + 1, bottleneck + 1), -1, dtype=np.double)
    choice_record = np.full((n_stream + 1, bottleneck + 1), -1, dtype=np.int64)
    dp_record[0][0] = 0
    for i in range(1, n_stream + 1):
        for j in range(min(i * (n_config - 1) + 1, bottleneck + 1)):
            for k in range(min(n_config, j + 1)):
                if dp_record[i - 1, j - k] + profit_matrix[k, i - 1] > dp_record[i][j]:
                    dp_record[i, j] = dp_record[i - 1, j - k] + profit_matrix[k, i - 1]
                    choice_record[i, j] = k
    choice = np.zeros(n_stream, dtype=np.int64)
    remain = bottleneck
    for i in range(n_stream, 0, -1):
        choice[i - 1] = choice_record[i, remain]
        remain -= choice_record[i, remain]
    return choice


def fake_replay(average_tpt):
    cfg = Config.fromfile('configs/custom/ssd.py')
    with open('datasets.json') as f:
        datasets = json.load(f)
    with open('tools/mmap_filter.pkl', 'rb') as f:
        mmap = pickle.load(f)
    mmap_filter = mmap[:, :-1, :]
    mmap_total = mmap[:, -1, :]
    n_config, n_epoch, n_stream = mmap_filter.shape
    if n_stream != len(datasets):
        print("Mismatch between dataset configuration file and dump file!")
        sys.exit(1)
    batch_size = list(datasets.values())[0]['size'] // n_epoch
    # bottleneck = average_tpt * n_stream
    baseline_map_filter, baseline_map_total = mmap_filter[average_tpt, :, :], mmap_total[average_tpt]
    aca_map_filter, aca_map_total = np.zeros(baseline_map_filter.shape), np.zeros(baseline_map_total.shape)
    results = [[] for _ in range(n_stream)]

    # Since epoch 0 have no distillation yet, start with even allocation
    choices = np.zeros((n_epoch, n_stream), dtype=np.int32)
    choices[0, :] = average_tpt

    previouses = []
    for name in datasets:
        with open(f'snapshot/result/{name}/000000.pkl', 'rb') as f:
            previous = pickle.load(f)
        previouses.append(previous)

    for epoch in range(n_epoch):
        print(f'Simulating epoch {epoch} ...')
        # collect result based on previous choice
        for name in datasets:
            stream = datasets[name]['id']
            stride = ORIGINAL_FRAMERATE // choices[epoch, stream]
            aca_map_filter[epoch, stream] = mmap_filter[choices[epoch, stream], epoch, stream]
            for i in range(epoch * batch_size, epoch * batch_size + batch_size):
                if i % stride == 0:
                    with open(f'snapshot/result/{name}/{i:06d}.pkl', 'rb') as f:
                        o = pickle.load(f)
                        results[stream].append(o)
                        previouses[stream] = o
                else:
                    results[stream].append(copy.deepcopy(previouses[stream]))
        # decide filter choice for next epoch
        if epoch + 1 < n_epoch:
            choices[epoch + 1, :] = choices[epoch, :]
    print('Simulation ended, starting evaluation ...')
    for name in datasets:
        stream = datasets[name]['id']
        cfg.data.test.ann_file = f'data/annotations/{name}.gt.json'
        cfg.data.test.img_prefix = ''
        dataset = build_dataset(cfg.data.test)
        print(f'Evaluting stream {stream} ...')
        aca_map_total[stream] = dataset.evaluate(results[stream], metric='bbox')['bbox_mAP']
    return baseline_map_filter, baseline_map_total, aca_map_filter, aca_map_total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fake replay')
    parser.add_argument('--throughput', '-t', type=int, default=3, help='average uplink throughput for each stream')
    args = parser.parse_args()
    baseline_map_distill, baseline_map_total, aca_map_distill, aca_map_total = fake_replay(args.throughput)
    print(f'baseline mAP = {baseline_map_total.sum()}, optimized mAP = {aca_map_total.sum()}')
