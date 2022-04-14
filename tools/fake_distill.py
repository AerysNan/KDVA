import ast
import sys
import json
import math
import copy
import numpy as np
import pickle
import argparse

from evaluate_from_file import evaluate_from_file
from multiprocessing import Pool

results = None


def choice_to_distill(choice):
    return f'{choice}'


def choice_to_filter(choice):
    return (choice + 1) * 5


def eval(name, stream, downsample):
    global results
    # define evaluation function for multiprocessing
    print(f'Evaluting stream {stream} ...')
    d = evaluate_from_file(results[stream], f'data/annotations/{name}.golden.json', downsample)
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


def fake_distill(throughput, fconfig, optimal, use_dp, n_stream, postfix, classwise, **_):
    with open('datasets.json') as f:
        datasets = json.load(f)
    with open(f'configs/cache/detrac_{postfix}.pkl', 'rb') as f:
        mmap = pickle.load(f)
    mmap_total, mmap_total_class = mmap['data'][:, fconfig, :n_stream, -1], mmap['classwise_data'][:, fconfig, :n_stream, -1]
    mmap_distill, mmap_distill_class = mmap['data'][:, fconfig, :n_stream, :-1], mmap['classwise_data'][:, fconfig, :n_stream, :-1]
    n_config, n_stream, n_epoch = mmap_distill_class.shape
    mmap_distill_gt, mmap_distill_class_gt = mmap_distill, mmap_distill_class
    if not optimal:
        with open(f'configs/cache/detrac_{postfix}_val.pkl', 'rb') as f:
            tmp = pickle.load(f)
            mmap_distill = np.zeros(mmap_distill.shape, dtype=np.double)
            mmap_distill_class = np.zeros(mmap_distill_class.shape, dtype=np.double)
            mmap_distill[:, :, 2:] = tmp['data'][:, fconfig, :n_stream, 1: -1]
            mmap_distill_class[:, :, 2:] = tmp['classwise_data'][:, fconfig, :n_stream, 1: -1]

    streams = []
    for i, stream in enumerate(datasets):
        if i >= n_stream:
            break
        streams.append(stream)
    if n_stream > len(datasets):
        print("Mismatch between dataset configuration file and dump file!")
        sys.exit(1)

    bottleneck = throughput * n_stream
    baseline_map, baseline_map_class = mmap_total[throughput], mmap_total_class[throughput]
    aca_map, aca_map_class = np.zeros(baseline_map.shape), np.zeros(baseline_map.shape)
    global results
    results = [[] for _ in range(n_stream)]

    # Since epoch 0 have no distillation yet, start with even allocation
    choices = np.zeros((n_epoch, n_stream), dtype=np.int64)
    choices[0, :] = throughput

    for epoch in range(n_epoch):
        # collect result based on previous choice
        for stream in range(n_stream):
            name = streams[stream]
            path = f'{name}_{choice_to_distill(choices[epoch, stream])}'
            if choices[epoch, stream] != 0:
                path += f'_{postfix}'
            with open(f'snapshot/result/{path}/{epoch:02d}.pkl', 'rb') as f:
                result = pickle.load(f)
            results[stream].extend(result)
        # decide distillation choice for next epoch
        print(f'Simulating epoch {epoch + 1} ...')
        if epoch + 1 < n_epoch:
            if epoch == 0 and not optimal:
                current_choice = throughput
            else:
                mmap_observation_epoch = mmap_distill_class[:, :, epoch + 1] if classwise else mmap_distill[:, :, epoch + 1]
                # start DP
                if optimal or use_dp:
                    current_choice = allocate(bottleneck, mmap_observation_epoch)
                else:
                    current_choice = copy.deepcopy(choices[epoch, :])
                    s = list(range(n_stream))
                    while len(s) > 1:
                        down_grad, up_grad = np.zeros(len(s), dtype=np.double), np.zeros(len(s), dtype=np.double)
                        for i in range(len(s)):
                            down_grad[i] = mmap_observation_epoch[current_choice[s[i]], i] - mmap_observation_epoch[current_choice[s[i]] -
                                                                                                                    1, i] if current_choice[s[i]] > 0 else mmap_observation_epoch[current_choice[s[i]], i]
                            up_grad[i] = down_grad[i] if current_choice[s[i]] < n_config - 1 else -1
                        thief, victim = np.argmax(up_grad), np.argmin(down_grad)
                        current_choice[s[thief]] += 1
                        current_choice[s[victim]] -= 1
                        s.remove(s[thief])

                    # current_choice = copy.deepcopy(choices[epoch, :])
                    # current_choice[:] = throughput
                    # for i in range(n_stream):
                    #     for j in range(n_stream):
                    #         if i == j:
                    #             continue
                    #         while True:
                    #             if current_choice[i] == n_config - 1 or current_choice[j] == 0:
                    #                 break
                    #             current_map = mmap_observation_epoch[current_choice[i], i] + mmap_observation_epoch[current_choice[j], j]
                    #             updated_map = mmap_observation_epoch[current_choice[i] + 1, i] + mmap_observation_epoch[current_choice[j]-1, j]
                    #             if current_map > updated_map:
                    #                 break
                    #             current_choice[i] += 1
                    #             current_choice[j] -= 1
            choices[epoch + 1, :] = current_choice
            mmap_gt_epoch = mmap_distill_class_gt[:, :, epoch + 1] if classwise else mmap_distill_gt[:, :, epoch + 1]
        print(f'Simulating epoch {epoch + 1} finished')
        print(f'mAP gt: {mmap_gt_epoch[current_choice, np.arange(n_stream)].mean():.3f} even gt: {mmap_gt_epoch[throughput, :].mean():.3f}')
        if epoch > 0 or optimal:
            print(f'mAP ob: {mmap_observation_epoch[current_choice, np.arange(n_stream)].mean():.3f} even ob: {mmap_observation_epoch[throughput, :].mean():.3f}')

    print('Simulation ended, starting evaluation ...')
    print(choices)
    pool, output = Pool(processes=6), {}

    for stream in range(n_stream):
        name = streams[stream]
        output[stream] = pool.apply_async(eval, (name, stream, (choice_to_filter(fconfig), 25),))
    pool.close()
    pool.join()
    for i in range(n_stream):
        result = output[i].get(600)
        if type(result) == tuple:
            result = result[0]
        aca_map[i] = result['bbox_mAP']
        classes_of_interest = ['car']
        mAPs_classwise = [result["classwise"][c] for c in classes_of_interest if not math.isnan(result["classwise"][c])]
        aca_map_class[i] = sum(mAPs_classwise) / len(mAPs_classwise)
    return baseline_map, baseline_map_class, aca_map, aca_map_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake distill')
    parser.add_argument('--throughput', '-t', type=int, default=3, help='average uplink throughput for each stream')
    parser.add_argument('--optimal', '-o', type=ast.literal_eval, default=True, help='use optimal knowledge')
    parser.add_argument('--use-dp', '-d', type=ast.literal_eval, default=True, help='use DP')
    parser.add_argument('--fconfig', '-f', type=int, default=4, help='inference configuration')
    parser.add_argument('--n-stream', '-n', type=int, default=4, help='number of streams')
    parser.add_argument('--postfix', '-p', type=str, default="short", help='dataset postfix')
    parser.add_argument('--classwise', '-c', type=ast.literal_eval, default=True, help='single class detection')
    args = parser.parse_args()
    baseline_map, baseline_map_class, aca_map, aca_map_class = fake_distill(**args.__dict__)
    print(f'baseline: {sum(baseline_map) / args.n_stream:.3f} classwise: {sum(baseline_map_class) / args.n_stream:.3f}')
    print(f'aca: {sum(aca_map) / args.n_stream:.3f} classwise: {sum(aca_map_class) / args.n_stream:.3f}')
