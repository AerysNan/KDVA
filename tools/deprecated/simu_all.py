import ast
import sys
import copy
import json
import math
import numpy as np
import pickle
import argparse

from multiprocessing import Pool
from tools.result_evaluate import generate_sample_position, evaluate_from_file

ORIGINAL_FRAMERATE = 60

results = None


def choice_to_distill(choice):
    return f'{choice}'


def choice_to_filter(choice):
    return (choice + 1) * 5


def eval(root, name, stream):
    global results
    # define evaluation function for multiprocessing
    print(f'Evaluting stream {stream} ...')
    d = evaluate_from_file(results[stream], f'{root}/data/annotations/{name}.golden.json')
    print(f'Evaluting stream {stream} finished!')
    return d


def allocate(bottleneck, profit_matrix):
    n_config, n_stream = profit_matrix.shape
    visited = set()
    # start DP
    dp_record = np.full((n_stream + 1, bottleneck + 1), -1, dtype=np.double)
    choice_record = np.full((n_stream + 1, bottleneck + 1), -1, dtype=np.int64)
    dp_record[0][0] = 0
    for i in range(1, n_stream + 1):
        for j in range(min(i * (n_config - 1) + 1, bottleneck + 1)):
            for k in range(min(n_config, j + 1)):
                visited.add((k, i - 1))
                if dp_record[i - 1, j - k] + profit_matrix[k, i - 1] > dp_record[i][j]:
                    dp_record[i, j] = dp_record[i - 1, j - k] + profit_matrix[k, i - 1]
                    choice_record[i, j] = k
    choice = np.zeros(n_stream, dtype=np.int64)
    remain = bottleneck
    for i in range(n_stream, 0, -1):
        choice[i - 1] = choice_record[i, remain]
        remain -= choice_record[i, remain]
    return choice, visited


def allocate2d(ret_btn, inf_btn, profit_matrix):
    n_uconfig, n_pconfig, n_stream = profit_matrix.shape
    if (profit_matrix == 0).all():
        return np.tile([ret_btn // n_stream, inf_btn // n_stream], (n_stream, 1)), set()
    # start DP
    visited = set()
    dp_record = np.full((n_stream + 1, ret_btn + 1, inf_btn + 1), -1, dtype=np.double)
    choice_record = np.full((n_stream + 1, ret_btn + 1, inf_btn + 1, 2), -1, dtype=np.int64)
    dp_record[0][0][0] = 0
    for i in range(1, n_stream + 1):
        for u in range(min(i * (n_uconfig - 1) + 1, ret_btn + 1)):
            for p in range(min(i * (n_pconfig - 1) + 1, inf_btn + 1)):
                for du in range(min(n_uconfig, u + 1)):
                    for dp in range(min(n_pconfig, p + 1)):
                        visited.add((du, dp, i - 1))
                        if dp_record[i - 1, u - du, p - dp] + profit_matrix[du, dp, i - 1] > dp_record[i, u, p]:
                            dp_record[i, u, p] = dp_record[i - 1, u - du, p - dp] + profit_matrix[du, dp, i - 1]
                            choice_record[i, u, p, :] = [du, dp]
    choice = np.zeros((n_stream, 2), dtype=np.int64)
    ret_remain, inf_remain = ret_btn, inf_btn
    for i in range(n_stream, 0, -1):
        choice[i - 1, :] = choice_record[i, ret_remain, inf_remain]
        du, dp = choice_record[i, ret_remain, inf_remain]
        ret_remain -= du
        inf_remain -= dp
    return choice, visited


def simulation(root, inf_tpt, ret_tpt, mode, n_stream, postfix, profile, classwise, estimate, **_):
    with open('datasets.json') as f:
        datasets = json.load(f)
    with open(f'configs/cache/detrac_{postfix}.pkl', 'rb') as f:
        mmap = pickle.load(f)
    mmap_total, mmap_total_class = mmap['data'][:, :, :n_stream, -1], mmap['classwise_data'][:, :, :n_stream, -1]
    mmap_by_epoch, mmap_by_epoch_class = mmap['data'][:, :, :n_stream, :-1], mmap['classwise_data'][:, :, :n_stream, :-1]
    n_rconfig, n_iconfig, n_stream, n_epoch = mmap_by_epoch_class.shape
    if profile is not None:
        with open(profile, 'rb') as f:
            tmp = pickle.load(f)
            mmap_profile = tmp['data'][:, :, :n_stream, :]
            mmap_profile_class = tmp['classwise_data'][:, :, :n_stream, :]
    streams = []
    for i, stream in enumerate(datasets):
        if i >= n_stream:
            break
        streams.append(stream)
    if n_stream > len(datasets):
        print("Mismatch between dataset configuration file and dump file!")
        sys.exit(1)

    inf_btn, ret_btn = inf_tpt * n_stream, ret_tpt * n_stream
    baseline_map_total, baseline_map_class = mmap_total[ret_tpt, inf_tpt, :], mmap_total_class[ret_tpt, inf_tpt, :]
    aca_map_total, aca_map_class = np.zeros(baseline_map_total.shape), np.zeros(baseline_map_class.shape)
    global results
    results = [[] for _ in range(n_stream)]

    # Since epoch 0 have no distillation yet, start with even allocation
    choices = np.zeros((n_epoch, n_stream, 2), dtype=np.int32)
    choices[0, :, :] = [ret_tpt, inf_tpt]
    total_probing_count = 0
    for epoch in range(n_epoch):
        # collect result based on previous choice
        if not estimate:
            for stream in range(n_stream):
                name = streams[stream]
                upload_cfg, filter_cfg = choices[epoch, stream]
                path = f'{name}_{choice_to_distill(upload_cfg)}_{postfix}'
                with open(f'{root}/snapshot/result/{path}/{epoch:02d}.pkl', 'rb') as f:
                    result = pickle.load(f)
                positions = generate_sample_position(filter_cfg + 1, 5)
                for start in range(0, len(result), 5):
                    for j in range(len(positions) - 1):
                        for k in range(positions[j] + 1, positions[j + 1]):
                            result[start + k] = result[start + positions[j]]
                    for k in range(positions[-1] + 1, 5):
                        result[start + k] = result[start + positions[-1]]
                results[stream].extend(result)
        # decide filter choice for next epoch
        print(f'Choosing configuration for epoch {epoch} ...')
        if epoch + 1 < n_epoch:
            if profile is None:
                mmap_obs_epoch = mmap_by_epoch_class[:, :, :, epoch + 1] if classwise else mmap_by_epoch[:, :, :, epoch + 1]
                current_choice, visited = allocate2d(ret_btn, inf_btn, mmap_obs_epoch)
                total_probing_count += len(visited)
                choices[epoch + 1, :, :] = current_choice
            else:
                mmap_obs_epoch = mmap_profile_class[:, :, :, epoch + 1] if classwise else mmap_profile[:, :, :, epoch + 1]
                if mode == 'sync':
                    # jointly allocation bandwidth and computation with optimal knowledge
                    current_choice, visited = allocate2d(ret_btn, inf_btn, mmap_obs_epoch)
                    total_probing_count += len(visited)
                    current_ret_choice = current_choice[:, 0]
                    current_inf_choice = current_choice[:, 1]
                elif mode == 'async':
                    mmap_obs_ret_epoch = mmap_obs_epoch[:, inf_tpt, :]
                    all_visited = set()
                    current_ret_choice, visited = allocate(ret_btn, mmap_obs_ret_epoch)
                    for (s, cfg) in visited:
                        all_visited.add((s, inf_tpt, cfg))
                    mmap_obs_inf_epoch = mmap_obs_epoch[ret_tpt, :, :]
                    current_inf_choice, visited = allocate(inf_btn, mmap_obs_inf_epoch)
                    for (s, cfg) in visited:
                        all_visited.add((s, cfg, ret_tpt))
                    total_probing_count += len(all_visited)
                elif mode == 'grad':
                    mmap_obs_ret_epoch = mmap_obs_epoch[:, inf_tpt, :]
                    current_ret_choice = copy.deepcopy(choices[epoch, :, 0])
                    current_ret_choice[:] = ret_tpt
                    visited = set()
                    for i in range(n_stream):
                        for j in range(n_stream):
                            if i == j:
                                continue
                            while True:
                                if current_ret_choice[i] == n_rconfig - 1 or current_ret_choice[j] == 0:
                                    break
                                visited.add((current_ret_choice[i], n_iconfig - 1, i))
                                visited.add((current_ret_choice[j], n_iconfig - 1, j))
                                visited.add((current_ret_choice[i] + 1, n_iconfig - 1, i))
                                visited.add((current_ret_choice[j] - 1, n_iconfig - 1, j))
                                current_map = mmap_obs_ret_epoch[current_ret_choice[i], i] + mmap_obs_ret_epoch[current_ret_choice[j], j]
                                updated_map = mmap_obs_ret_epoch[current_ret_choice[i] + 1,  i] + mmap_obs_ret_epoch[current_ret_choice[j] - 1,  j]
                                if current_map > updated_map:
                                    break
                                current_ret_choice[i] += 1
                                current_ret_choice[j] -= 1
                    mmap_obs_inf_epoch = mmap_obs_epoch[ret_tpt, :, :]
                    current_inf_choice = copy.deepcopy(choices[epoch, :, 1])
                    current_inf_choice[:] = inf_tpt
                    for i in range(n_stream):
                        for j in range(n_stream):
                            if i == j:
                                continue
                            while True:
                                if current_inf_choice[i] == n_iconfig - 1 or current_inf_choice[j] == 0:
                                    break
                                visited.add((current_ret_choice[i], current_inf_choice[i], i))
                                visited.add((current_ret_choice[j], current_inf_choice[j], j))
                                visited.add((current_ret_choice[i], current_inf_choice[i] + 1, i))
                                visited.add((current_ret_choice[j], current_inf_choice[j] - 1, j))
                                current_map = mmap_obs_inf_epoch[current_inf_choice[i], i] + mmap_obs_inf_epoch[current_inf_choice[j], j]
                                updated_map = mmap_obs_inf_epoch[current_inf_choice[i] + 1, i] + \
                                    mmap_obs_inf_epoch[current_inf_choice[j] - 1, j]
                                if current_map > updated_map:
                                    break
                                current_inf_choice[i] += 1
                                current_inf_choice[j] -= 1
                    total_probing_count += len(visited)
                else:
                    print('Invalid allocation plan!')
                    sys.exit(1)
                if epoch + 1 == 1:
                    choices[1, :, 0] = ret_tpt
                if epoch + 1 < n_epoch - 1:
                    choices[epoch + 2, :, 0] = current_ret_choice
                choices[epoch + 1, :, 1] = current_inf_choice
            mmap_gt_epoch = mmap_by_epoch_class[:, :, :, epoch + 1] if classwise else mmap_by_epoch[:, :, :, epoch + 1]
            print(f'Configuration for epoch {epoch + 1} chosen')
            print(f'mAP gt: {mmap_gt_epoch[choices[epoch + 1, :, 0], choices[epoch + 1, :, 1], np.arange(n_stream)].mean():.3f} even gt: {mmap_gt_epoch[ret_tpt, inf_tpt, :].mean():.3f}')
            print(f'mAP ob: {mmap_obs_epoch[choices[epoch + 1, :, 0], choices[epoch + 1, :, 1], np.arange(n_stream)].mean():.3f} even ob: {mmap_obs_epoch[ret_tpt, inf_tpt, :].mean():.3f}')

    print('Simulation ended, starting evaluation ...')
    print(choices)
    if estimate:
        est_map, est_map_class = np.zeros((n_epoch, n_stream), dtype=np.double), np.zeros((n_epoch, n_stream), dtype=np.double)
        for i in range(n_epoch):
            est_map[i, :] = mmap_by_epoch[choices[i, :, 0], choices[i, :, 1], np.arange(n_stream), i]
            est_map_class[i, :] = mmap_by_epoch_class[choices[i, :, 0], choices[i, :, 1], np.arange(n_stream), i]
        return np.average(mmap_by_epoch[ret_tpt, inf_tpt, :, :], axis=1), np.average(mmap_by_epoch_class[ret_tpt, inf_tpt, :, :], axis=1), np.average(est_map, axis=0), np.average(est_map_class, axis=0), total_probing_count

    pool, output = Pool(processes=6), {}
    for stream in range(n_stream):
        name = streams[stream]
        output[stream] = pool.apply_async(eval, (root, name, stream,))
    pool.close()
    pool.join()

    for i in range(n_stream):
        result = output[i].get()
        if type(result) == tuple:
            result = result[0]
        aca_map_total[i] = result[-1]['bbox_mAP']
        classes_of_interest = ['car']
        mAPs_classwise = [result[-1]["classwise"][c] for c in classes_of_interest if not math.isnan(result[-1]["classwise"][c])]
        aca_map_class[i] = sum(mAPs_classwise) / len(mAPs_classwise)
    return baseline_map_total, baseline_map_class, aca_map_total, aca_map_class, total_probing_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake replay')
    parser.add_argument('--root', '-r', type=str, required=True, help='data root')
    parser.add_argument('--inf-tpt', type=int, default=3, help='average inference throughput for each stream')
    parser.add_argument('--ret-tpt', type=int, default=3, help='average uplink throughput for each stream')
    parser.add_argument('--mode', '-m', type=str, default='sync', help='allocaiton mode')
    parser.add_argument('--n-stream', '-n', type=int, default=4, help='number of streams')
    parser.add_argument('--profile', '-i', type=str, default=None, help='accuracy profile')
    parser.add_argument('--postfix', '-p', type=str, default="short", help='dataset postfix')
    parser.add_argument('--classwise', '-c', type=ast.literal_eval, default=True, help='single class detection')
    parser.add_argument('--estimate', '-e', type=ast.literal_eval, default=True, help='estimate mAP')
    args = parser.parse_args()
    baseline_map_total, baseline_map_class, aca_map_total, aca_map_classwise, probing_count = simulation(**args.__dict__)
    print(f'baseline: {sum(baseline_map_total) / args.n_stream:.3f} classwise: {sum(baseline_map_class) / args.n_stream:.3f}')
    print(f'aca: {sum(aca_map_total) / args.n_stream:.3f} classwise: {sum(aca_map_classwise) / args.n_stream:.3f}')
    print(f'probing count: {probing_count}')
