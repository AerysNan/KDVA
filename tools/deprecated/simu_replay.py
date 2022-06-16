import ast
import math
import copy
import json
import numpy as np
import pickle
import argparse

from tools.deprecated.replay_trace import replay_trace
from multiprocessing import Pool

results = None


def eval(root, path, name, framerate, batch_size):
    print(f"Evaluating {name}...")
    d = replay_trace(root, path, name, framerate, batch_size, summarize=True)
    print(f"Evaluating {name} finished!")
    return d


def choice_to_framerate(choice):
    return (choice + 1) * 5


def choice_to_distill(choice):
    return f'{choice}'


def allocate(bottleneck, profit_matrix):
    n_config, n_stream = profit_matrix.shape
    if (profit_matrix == 0).all():
        return np.repeat(bottleneck // n_stream, n_stream), set()
    # start DP
    dp_record = np.full((n_stream + 1, bottleneck + 1), -1, dtype=np.double)
    choice_record = np.full((n_stream + 1, bottleneck + 1), -1, dtype=np.int64)
    dp_record[0][0] = 0
    visited = set()
    for i in range(1, n_stream + 1):
        for j in range(bottleneck + 1):
            for k in range(min(n_config, j + 1)):
                visited.add((k, i - 1))
                if dp_record[i - 1, j - k] + profit_matrix[k, i - 1] > dp_record[i, j]:
                    dp_record[i, j] = dp_record[i - 1, j - k] + profit_matrix[k, i - 1]
                    choice_record[i, j] = k
    choice = np.zeros(n_stream, dtype=np.int64)
    remain = bottleneck
    for i in range(n_stream, 0, -1):
        choice[i - 1] = choice_record[i, remain]
        remain -= choice_record[i, remain]
    return choice, visited


def hc(bottleneck, profit_matrix):
    visited = set()
    n_config, n_stream = profit_matrix.shape
    gradient, choices = np.zeros(n_stream, dtype=np.double), np.zeros(n_stream, dtype=np.int64)

    def grad(choices, i):
        if choices[i] == n_config - 1:
            return -1
        else:
            visited.add((choices[i], i))
            visited.add((choices[i] + 1, i))
            return profit_matrix[choices[i] + 1, i] - profit_matrix[choices[i], i]

    for i in range(n_stream):
        gradient[i] = grad(choices, i)
    while bottleneck > 0:
        indices = np.argsort(gradient)
        stream = indices[-1]
        if gradient[stream] == -1:
            break
        choices[stream] += 1
        bottleneck -= 1
        gradient[stream] = grad(choices, stream)
    return choices, visited


def fake_replay(root, throughput, rconfig, mode, n_stream, postfix, profile, classwise, estimate, **_):
    with open('datasets.json') as f:
        datasets = json.load(f)
    with open(f'configs/cache/detrac_{postfix}.pkl', 'rb') as f:
        mmap = pickle.load(f)
    mmap_total, mmap_total_class = mmap['data'][rconfig, :, :n_stream, -1], mmap['classwise_data'][rconfig, :, :n_stream, -1]
    mmap_by_epoch, mmap_by_epoch_class = mmap['data'][rconfig, :, :n_stream, :-1], mmap['classwise_data'][rconfig, :, :n_stream, :-1]
    n_config, n_stream, n_epoch = mmap_by_epoch_class.shape
    if profile is not None:
        with open(profile, 'rb') as f:
            tmp = pickle.load(f)
            mmap_profile = tmp['data'][rconfig, :, :n_stream, :]
            mmap_profile_class = tmp['classwise_data'][rconfig, :, :n_stream, :]
    streams = []
    for i, stream in enumerate(datasets):
        if i >= n_stream:
            break
        streams.append(stream)
    bottleneck = throughput * n_stream
    baseline_map_total, baseline_map_class = mmap_total[throughput, :], mmap_total_class[throughput, :]
    aca_map_total, aca_map_classwise = np.zeros(baseline_map_total.shape), np.zeros(baseline_map_total.shape)
    global results
    results = [[] for _ in range(n_stream)]

    # Since epoch 0 have no distillation yet, start with even allocation
    choices = np.zeros((n_epoch, n_stream), dtype=np.int32)
    choices[0, :] = throughput
    total_probing_count = 0

    for epoch in range(n_epoch):
        print(f'Choosing configuration for epoch {epoch + 1} ...')
        # decide filter choice for next epoch
        if epoch + 1 < n_epoch:
            if profile is None:
                mmap_obs_epoch = mmap_by_epoch_class[:, :, epoch + 1] if classwise else mmap_by_epoch[:, :, epoch + 1]
            else:
                mmap_obs_epoch = mmap_profile_class[:, :, epoch + 1] if classwise else mmap_profile[:, :, epoch + 1]
            if mode == 'dp':
                # start DP
                current_choice, visited = allocate(bottleneck, mmap_obs_epoch)
                total_probing_count += len(visited)
            elif mode == 'hc':
                current_choice, visited = hc(bottleneck, mmap_obs_epoch)
                total_probing_count += len(visited)
            else:
                # current_choice = copy.deepcopy(choices[epoch, :])
                # s = list(range(n_stream))
                # while len(s) > 1:
                #     down_grad, up_grad = np.zeros(len(s), dtype=np.double), np.zeros(len(s), dtype=np.double)
                #     for i in range(len(s)):
                #         if current_choice[s[i]] == 0:
                #             down_grad[i] = 1
                #             up_grad[i] = mmap_observation_epoch[current_choice[s[i]], s[i]]
                #         elif current_choice[s[i]] == n_config - 1:
                #             down_grad = mmap_observation_epoch[current_choice[s[i]], s[i]] - mmap_observation_epoch[current_choice[s[i]] - 1, s[i]]
                #             up_grad[i] = -1
                #         else:
                #             down_grad[i] = mmap_observation_epoch[current_choice[s[i]], s[i]] - mmap_observation_epoch[current_choice[s[i]] - 1, s[i]]
                #             up_grad[i] = mmap_observation_epoch[current_choice[s[i]], s[i]] - mmap_observation_epoch[current_choice[s[i]] - 1, s[i]]
                #     thief, victim = np.argmax(up_grad), np.argmin(down_grad)
                #     current_choice[s[thief]] += 1
                #     current_choice[s[victim]] -= 1
                #     s.remove(s[thief])

                current_choice = copy.deepcopy(choices[epoch, :])
                current_choice[:] = throughput
                visited = set()
                for i in range(n_stream):
                    for j in range(n_stream):
                        if i == j:
                            continue
                        while True:
                            if current_choice[i] == n_config - 1 or current_choice[j] == 0:
                                break
                            visited.add((current_choice[i], i))
                            visited.add((current_choice[j], j))
                            visited.add((current_choice[i] + 1, i))
                            visited.add((current_choice[j] - 1, j))
                            current_map = mmap_obs_epoch[current_choice[i], i] + mmap_obs_epoch[current_choice[j], j]
                            updated_map = mmap_obs_epoch[current_choice[i] + 1, i] + mmap_obs_epoch[current_choice[j] - 1, j]
                            if current_map > updated_map:
                                break
                            current_choice[i] += 1
                            current_choice[j] -= 1
                total_probing_count += len(visited)

            choices[epoch + 1, :] = current_choice
            mmap_gt_epoch = mmap_by_epoch_class[:, :, epoch + 1] if classwise else mmap_by_epoch[:, :, epoch + 1]
            print(f'Configuration for epoch {epoch + 1} chosen')
            print(f'mAP gt: {mmap_gt_epoch[current_choice, np.arange(n_stream)].mean():.3f} even gt: {mmap_gt_epoch[throughput, :].mean():.3f}')
            print(f'mAP ob: {mmap_obs_epoch[current_choice, np.arange(n_stream)].mean():.3f} even ob: {mmap_obs_epoch[throughput, :].mean():.3f}')

    print('Simulation ended, starting evaluation ...')
    print(choices)

    if estimate:
        est_map, est_map_class = np.zeros((n_epoch, n_stream), dtype=np.double), np.zeros((n_epoch, n_stream), dtype=np.double)
        for i in range(n_epoch):
            est_map[i, :] = mmap_by_epoch[choices[i, :], np.arange(n_stream), i]
            est_map_class[i, :] = mmap_by_epoch_class[choices[i, :], np.arange(n_stream), i]
        return np.average(mmap_by_epoch[throughput, :, :], axis=1), np.average(mmap_by_epoch_class[throughput, :, :], axis=1), np.average(est_map, axis=0), np.average(est_map_class, axis=0), total_probing_count

    pool, output = Pool(processes=6), {}

    for stream in range(n_stream):
        name = streams[stream]
        path = f'snapshot/result/{name}_{choice_to_distill(rconfig)}_{postfix}'
        output[stream] = pool.apply_async(eval, (root, path, name, choice_to_framerate(choices[:, stream]), 500,))
    pool.close()
    pool.join()
    for i in range(n_stream):
        result = output[i].get()
        if type(result) == tuple:
            result = result[0]
        aca_map_total[i] = result[-1]['bbox_mAP']
        classes_of_interest = ['car']
        mAPs_classwise = [result[-1]["classwise"][c] for c in classes_of_interest if not math.isnan(result[-1]["classwise"][c])]
        aca_map_classwise[i] = sum(mAPs_classwise) / len(mAPs_classwise)
    return baseline_map_total, baseline_map_class, aca_map_total, aca_map_classwise, total_probing_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake replay')
    parser.add_argument('--root', '-r', type=str, required=True, help='data root')
    parser.add_argument('--throughput', '-t', type=int, default=3, help='average inference throughput for each stream')
    parser.add_argument('--mode', '-m', type=str, default='dp', help='allocation mode')
    parser.add_argument('--profile', '-i', type=str, default=None, help='accuracy profile')
    parser.add_argument('--rconfig', '-d', type=int, default=3, help='level of retraining')
    parser.add_argument('--n-stream', '-n', type=int, default=4, help='number of streams')
    parser.add_argument('--postfix', '-p', type=str, default="short", help='dataset postfix')
    parser.add_argument('--classwise', '-c', type=ast.literal_eval, default=True, help='single class detection')
    parser.add_argument('--estimate', '-e', type=ast.literal_eval, default=True, help='estimate mAP')
    args = parser.parse_args()
    baseline_map_total, baseline_map_class, aca_map_total, aca_map_classwise, probing_count = fake_replay(**args.__dict__)
    print(f'baseline: {sum(baseline_map_total) / args.n_stream:.3f} classwise: {sum(baseline_map_class) / args.n_stream:.3f}')
    print(f'aca: {sum(aca_map_total) / args.n_stream:.3f} classwise: {sum(aca_map_classwise) / args.n_stream:.3f}')
    print(f'probing count: {probing_count}')
