import ast
import sys
import copy
import json
import numpy as np
import pickle
import argparse

from replay_trace import replay_trace
from multiprocessing import Pool

results = None


def eval(path, name, framerate, batch_size):
    print(f"Evaluating {name}...")
    d = replay_trace(path, name, framerate, batch_size)
    print(f"Evaluating {name} finished!")
    return d


def choice_to_framerate(choice):
    return (choice + 1) * 100


def choice_to_distill(choice):
    return choice * 20


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


def fake_replay(average_tpt, optimal, use_dp, size, dconfig, n_stream, postfix):
    with open('datasets.json') as f:
        datasets = json.load(f)
    with open(f'configs/cache/map_all_golden_na.pkl', 'rb') as f:
        mmap = pickle.load(f)[dconfig, :, :n_stream, :]
    with open(f'configs/cache/map_all_class_golden_na.pkl', 'rb') as f:
        mmap_class = pickle.load(f)[dconfig, :, :n_stream, :]
    mmap_total = mmap[:, :, -1]
    mmap_filter = mmap_class[:, :, :-1]
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
    baseline_map_total, baseline_map_class = mmap_total[average_tpt], mmap_class[average_tpt, :, -1]
    aca_map_total, aca_map_classwise = np.zeros(baseline_map_total.shape), np.zeros(baseline_map_total.shape)
    global results
    results = [[] for _ in range(n_stream)]

    # Since epoch 0 have no distillation yet, start with even allocation
    choices = np.zeros((n_epoch, n_stream), dtype=np.int32)
    choices[0, :] = average_tpt

    for epoch in range(n_epoch):
        print(f'Simulating epoch {epoch} ...')
        # decide filter choice for next epoch
        if epoch + 1 < n_epoch:
            if optimal:
                mmap_observation_epoch = mmap_filter[:, :, epoch + 1]
            else:
                mmap_observation_epoch = mmap_filter[:, :, epoch]
            if use_dp or optimal:
                # start DP
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
                            updated_map = mmap_observation_epoch[current_choice[i] + 1, i] + mmap_observation_epoch[current_choice[j] - 1, j]
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
        path = f'snapshot/result/{name}_{choice_to_distill(dconfig):03d}'
        if dconfig > 0:
            path += '_na'
        output.append(pool.apply_async(eval, (path, name, choice_to_framerate(choices[:, stream]), size,)))
    pool.close()
    pool.join()
    for i in range(n_stream):
        result = output[i].get()
        aca_map_total[i] = result[-1]['bbox_mAP']
        aca_map_classwise[i] = result[-1]['bbox_mAP_car']
    return baseline_map_total, baseline_map_class, aca_map_total, aca_map_classwise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake replay')
    parser.add_argument('--throughput', '-t', type=int, default=3, help='average uplink throughput for each stream')
    parser.add_argument('--optimal', '-o', type=ast.literal_eval, default=True, help='use optimal knowledge')
    parser.add_argument('--use-dp', '-a', type=ast.literal_eval, default=True, help='use DP')
    parser.add_argument('--distill', '-d', type=int, default=3, help='level of distillation')
    parser.add_argument('--stream', '-n', type=int, default=4, help='number of streams')
    parser.add_argument('--size', '-s', type=int, default=500, help='epoch size')
    parser.add_argument('--postfix', '-p', type=str, default="500_all_acc", help='postfix')
    args = parser.parse_args()
    baseline_map_total, baseline_map_class, aca_map_total, aca_map_classwise = fake_replay(args.throughput, args.optimal, args.use_dp, args.size, args.distill, args.stream, args.postfix)
    print(f'baseline: {sum(baseline_map_total) / args.stream:.3f} classwise: {sum(baseline_map_class) / args.stream:.3f}')
    print(f'aca: {sum(aca_map_total) / args.stream:.3f} classwise: {sum(aca_map_classwise) / args.stream:.3f}')
