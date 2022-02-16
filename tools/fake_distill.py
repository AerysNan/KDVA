import ast
import sys
import json
import copy
import numpy as np
import pickle
import argparse

from mmcv import Config
from mmdet.datasets import build_dataset
from multiprocessing import Pool


results = None


def choice_to_distill(choice):
    return choice * 20


def eval(cfg, stream):
    global results
    # define evaluation function for multiprocessing
    print(f'Evaluting stream {stream} ...')
    dataset = build_dataset(cfg.data.test)
    d = {
        'id': stream,
        'mAP': dataset.evaluate(results[stream], metric='bbox', classwise=True),
    }
    print(f'Evaluting stream {stream} finished!')
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


def fake_distill(average_tpt, optimal, use_dp, n_stream, postfix):
    with open('datasets.json') as f:
        datasets = json.load(f)
    with open(f'configs/cache/map_distill.pkl', 'rb') as f:
        mmap = pickle.load(f)[:, :n_stream, :]
    mmap_total = mmap[:, :, -1]
    mmap_distill = mmap[:, :, :-1]
    n_config, n_stream, n_epoch = mmap_distill.shape

    if not optimal:
        mmap_distill = np.concatenate((np.zeros((n_config, n_stream, 2)), mmap_distill[:, :, 1: -1]), axis=2)
    streams = []
    for i, stream in enumerate(datasets):
        if i >= n_stream:
            break
        streams.append(stream)
    if n_stream > len(datasets):
        print("Mismatch between dataset configuration file and dump file!")
        sys.exit(1)

    bottleneck = average_tpt * n_stream
    baseline_map_total = mmap_total[average_tpt]
    aca_map_total = np.zeros(baseline_map_total.shape)
    aca_map_class = np.zeros(baseline_map_total.shape)
    global results
    results = [[] for _ in range(n_stream)]

    # Since epoch 0 have no distillation yet, start with even allocation
    choices = np.zeros((n_epoch, n_stream), dtype=np.int64)
    choices[0, :] = average_tpt

    for epoch in range(n_epoch):
        # print(f'Simulating epoch {epoch} ...')
        # collect result based on previous choice
        for stream in range(n_stream):
            name = streams[stream]
            with open(f'snapshot/result/{name}_{choice_to_distill(choices[epoch, stream]):03d}-{postfix}/{epoch:02d}.pkl', 'rb') as f:
                result = pickle.load(f)
            results[stream].extend(result)
        # decide distillation choice for next epoch
        if epoch + 1 < n_epoch:
            if epoch == 0 and not optimal:
                choices[epoch + 1, :] = average_tpt
            else:
                mmap_observation_epoch = mmap_distill[:, :, epoch + 1]
                # start DP
                if optimal or use_dp:
                    current_choice = allocate(bottleneck, mmap_observation_epoch)
                else:
                    current_choice = copy.deepcopy(choices[epoch, :])
                    current_choice[:] = average_tpt
                    for i in range(n_stream):
                        for j in range(n_stream):
                            if i == j:
                                continue
                            while True:
                                if current_choice[i] == n_config - 1 or current_choice[j] == 0:
                                    break
                                current_map = mmap_observation_epoch[current_choice[i], i] + mmap_observation_epoch[current_choice[j], j]
                                updated_map = mmap_observation_epoch[current_choice[i] + 1, i] + mmap_observation_epoch[current_choice[j]-1, j]
                                if current_map > updated_map:
                                    break
                                current_choice[i] += 1
                                current_choice[j] -= 1
                choices[epoch + 1, :] = current_choice
    print('Simulation ended, starting evaluation ...')
    print(choices)
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
        try:
            result = output[i].get(600)
            aca_map_total[result['id']] = result['mAP']['bbox_mAP']
            aca_map_class[result['id']] = result['mAP']['classwise'][2][1]
        except:
            print('Timeout occurred!')
    return baseline_map_total, aca_map_total, aca_map_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake distill')
    parser.add_argument('--throughput', '-t', type=int, default=3, help='average uplink throughput for each stream')
    parser.add_argument('--optimal', '-o', type=ast.literal_eval, default=True, help='use optimal knowledge')
    parser.add_argument('--use-dp', '-d', type=ast.literal_eval, default=False, help='use DP')
    parser.add_argument('--stream', '-n', type=int, default=4, help='number of streams')
    parser.add_argument('--postfix', '-p', type=str, default="500_all_acc", help='dataset postfix')
    args = parser.parse_args()
    baseline_map_total, aca_map_total, aca_map_class = fake_distill(args.throughput, args.optimal, args.use_dp, args.stream, args.postfix)
    print('baseline')
    print(f'{sum(baseline_map_total) / args.stream:.3f}')
    print('aca')
    print(f'{sum(aca_map_total) / args.stream:.3f}')
    print('classwise')
    print(f'{sum(aca_map_class) / args.stream:.3f}')
