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
    return f'{choice * 20:03d}'


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


def fake_distill(throughput, optimal, use_dp, n_stream, postfix, classwise, **_):
    with open('datasets.json') as f:
        datasets = json.load(f)
    with open(f'configs/cache/map_distill_golden_{postfix}.pkl', 'rb') as f:
        mmap = pickle.load(f)[:, :n_stream, :]
    with open(f'configs/cache/map_distill_class_golden_{postfix}.pkl', 'rb') as f:
        mmap_class = pickle.load(f)[:, :n_stream, :]
    mmap_total, mmap_total_class = mmap[:, :, -1], mmap_class[:, :, -1]
    mmap_distill, mmap_distill_class = mmap[:, :, :-1], mmap_class[:, :, :-1]
    n_config, n_stream, n_epoch = mmap_distill_class.shape

    if not optimal:
        with open(f'configs/cache/map_distill_class_val_{postfix}.pkl', 'rb') as f:
            mmap_distill_class = pickle.load(f)[:, :n_stream, :]
            # mmap_distill_class = np.concatenate((np.zeros((n_config, n_stream, 1)), mmap_distill_class[:, :, : -1]), axis=2)
        with open(f'configs/cache/map_distill_val_{postfix}.pkl', 'rb') as f:
            mmap_distill = pickle.load(f)[:, :n_stream, :]
            # mmap_distill = np.concatenate((np.zeros((n_config, n_stream, 1)), mmap_distill_class[:, :, : -1]), axis=2)

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
        # print(f'Simulating epoch {epoch} ...')
        # collect result based on previous choice
        for stream in range(n_stream):
            name = streams[stream]
            path = f'{name}_{choice_to_distill(choices[epoch, stream])}'
            if '000' not in path:
                path += f'_{postfix}'
            with open(f'snapshot/result/{path}/{epoch:02d}.pkl', 'rb') as f:
                result = pickle.load(f)
            results[stream].extend(result)
        # decide distillation choice for next epoch
        if epoch + 1 < n_epoch:
            if epoch == 0 and not optimal:
                choices[epoch + 1, :] = throughput
            else:
                mmap_observation_epoch = mmap_distill_class[:, :, epoch + 1] if classwise else mmap_distill[:, :, epoch + 1]
                # start DP
                if optimal or use_dp:
                    current_choice = allocate(bottleneck, mmap_observation_epoch)
                else:
                    current_choice = copy.deepcopy(choices[epoch, :])
                    current_choice[:] = throughput
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
        cfg = Config.fromfile('configs/custom/ssd_base.py')
        cfg.data.test.ann_file = f'data/annotations/{name}.golden.json'
        cfg.data.test.img_prefix = ''
        output.append(pool.apply_async(eval, (cfg, stream,)))
    pool.close()
    pool.join()
    for i in range(n_stream):
        try:
            result = output[i].get(600)
            aca_map[result['id']] = result['mAP']['bbox_mAP']
            aca_map_class[result['id']] = result['mAP']["bbox_mAP_car"]
        except:
            print('Timeout occurred!')
    return baseline_map, baseline_map_class, aca_map, aca_map_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake distill')
    parser.add_argument('--throughput', '-t', type=int, default=3, help='average uplink throughput for each stream')
    parser.add_argument('--optimal', '-o', type=ast.literal_eval, default=True, help='use optimal knowledge')
    parser.add_argument('--use-dp', '-d', type=ast.literal_eval, default=True, help='use DP')
    parser.add_argument('--n-stream', '-n', type=int, default=4, help='number of streams')
    parser.add_argument('--postfix', '-p', type=str, default="short", help='dataset postfix')
    parser.add_argument('--classwise', '-c', type=ast.literal_eval, default=True, help='single class detection')
    args = parser.parse_args()
    baseline_map, baseline_map_class, aca_map, aca_map_class = fake_distill(**args.__dict__)
    print(f'baseline: {sum(baseline_map) / args.n_stream:.3f} classwise: {sum(baseline_map_class) / args.n_stream:.3f}')
    print(f'aca: {sum(aca_map) / args.n_stream:.3f} classwise: {sum(aca_map_class) / args.n_stream:.3f}')
