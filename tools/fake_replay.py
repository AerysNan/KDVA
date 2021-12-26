import ast
import sys
import copy
import json
import numpy as np
import pickle
import argparse

from multiprocessing import Pool
from mmcv import Config
from mmdet.datasets import build_dataset

ORIGINAL_FRAMERATE = 60

results = None


def eval(cfg, stream):
    global results
    # define evaluation function for multiprocessing
    # print(f'Evaluting stream {stream} ...')
    dataset = build_dataset(cfg.data.test)
    d = {
        'id': stream,
        'mAP': dataset.evaluate(results[stream], metric='bbox')['bbox_mAP'],
    }
    # print(f'Evaluting stream {stream} finished!')
    return d


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


def fake_replay(average_tpt, optimal, use_dp, size, dconfig, n_stream):
    cfg = Config.fromfile('configs/custom/ssd.py')
    with open('datasets.json') as f:
        datasets = json.load(f)
    with open(f'configs/cache/map_distill_filter_{size}.pkl', 'rb') as f:
        mmap = pickle.load(f)[dconfig, :, :n_stream, :]
    mmap_total = mmap[:, :, -1]
    mmap_filter = mmap[:, :, :-1]
    n_config, n_stream, n_epoch = mmap_filter.shape
    if n_stream > len(datasets):
        print("Mismatch between dataset configuration file and dump file!")
        sys.exit(1)
    streams = []
    for i, stream in enumerate(datasets):
        if i >= n_stream:
            break
        streams.append(stream)
    batch_size = list(datasets.values())[0]['size'] // n_epoch
    if batch_size != size:
        print("Mismatch between dataset configuration file and dump file!")
        sys.exit(1)
    bottleneck = average_tpt * n_stream
    baseline_map_total = mmap_total[average_tpt - 1]
    aca_map_total = np.zeros(baseline_map_total.shape)
    global results
    results = [[] for _ in range(n_stream)]

    # Since epoch 0 have no distillation yet, start with even allocation
    choices = np.zeros((n_epoch, n_stream), dtype=np.int32)
    choices[0, :] = average_tpt

    previouses = []
    for stream in range(n_stream):
        name = streams[stream]
        with open(f'snapshot/result/{name}_{dconfig}-{batch_size}/000000.pkl', 'rb') as f:
            previous = pickle.load(f)
        previouses.append(previous)

    for epoch in range(n_epoch):
        print(f'Simulating epoch {epoch} ...')
        # collect result based on previous choice
        for stream in range(n_stream):
            name = streams[stream]
            stride = ORIGINAL_FRAMERATE // choices[epoch, stream]
            for i in range(epoch * batch_size, epoch * batch_size + batch_size):
                if i % stride == 0:
                    with open(f'snapshot/result/{name}_{dconfig}-{batch_size}/{i:06d}.pkl', 'rb') as f:
                        o = pickle.load(f)
                        results[stream].append(o)
                        previouses[stream] = o
                else:
                    results[stream].append(copy.deepcopy(previouses[stream]))
        # decide filter choice for next epoch
        if epoch + 1 < n_epoch:
            if optimal:
                mmap_observation_epoch = mmap_filter[:, :, epoch + 1]
            else:
                mmap_observation_epoch = mmap_filter[:, :, epoch]
            if use_dp:
                # start DP
                current_choice = allocate(bottleneck - n_stream, mmap_observation_epoch)
                current_choice += 1
            else:
                current_choice = copy.deepcopy(choices[epoch, :])
                for _ in range(n_stream):
                    gradient = np.zeros(n_stream, dtype=np.double)
                    for stream in range(n_stream):
                        if current_choice[stream] == 1:
                            gradient[stream] = mmap_observation_epoch[current_choice[stream] - 1, stream]
                        else:
                            gradient[stream] = mmap_observation_epoch[current_choice[stream] - 1, stream] - mmap_observation_epoch[current_choice[stream] - 2, stream]
                    index = np.argsort(gradient)
                    thief, victim = n_stream - 1, 0
                    while thief >= 0 and current_choice[index[thief]] == n_config:
                        thief -= 1
                    while victim < n_stream and current_choice[index[victim]] == 1:
                        victim += 1
                    if thief > victim:
                        current_choice[index[thief]] += 1
                        current_choice[index[victim]] -= 1
                    else:
                        break
            choices[epoch + 1, :] = current_choice

    print('Simulation ended, starting evaluation ...')
    pool, output = Pool(processes=4), []

    for stream in range(n_stream):
        name = streams[stream]
        cfg = Config.fromfile('configs/custom/ssd.py')
        cfg.data.test.ann_file = f'data/annotations/{name}.gt.json'
        cfg.data.test.img_prefix = ''
        output.append(pool.apply_async(eval, (cfg, stream,)))
    pool.close()
    pool.join()
    for i in range(n_stream):
        result = output[i].get()
        aca_map_total[result['id']] = result['mAP']
    return baseline_map_total, aca_map_total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake replay')
    parser.add_argument('--throughput', '-t', type=int, default=3, help='average uplink throughput for each stream')
    parser.add_argument('--optimal', '-o', type=ast.literal_eval, default=True, help='use optimal knowledge')
    parser.add_argument('--aggresive', '-a', type=ast.literal_eval, default=True, help='use DP')
    parser.add_argument('--distill', '-d', type=int, default=3, help='level of distillation')
    parser.add_argument('--stream', '-n', type=int, default=4, help='number of streams')
    parser.add_argument('--size', '-s', type=int, default=600, help='epoch size')
    args = parser.parse_args()
    baseline_map_total, aca_map_total = fake_replay(args.throughput, args.optimal, args.aggresive, args.size, args.distill, args.stream)
    print('baseline')
    for v in baseline_map_total:
        print(v)
    print('aca')
    for v in aca_map_total:
        print(v)
