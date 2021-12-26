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
    print(f'Evaluting stream {stream} ...')
    dataset = build_dataset(cfg.data.test)
    d = {
        'id': stream,
        'mAP': dataset.evaluate(results[stream], metric='bbox')['bbox_mAP'],
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


def allocate2d(upload_btn, process_btn, profit_matrix):
    n_uconfig, n_pconfig, n_stream = profit_matrix.shape
    # start DP
    dp_record = np.full((n_stream + 1, upload_btn + 1, process_btn + 1), -1, dtype=np.double)
    choice_record = np.full((n_stream + 1, upload_btn + 1, process_btn + 1, 2), -1, dtype=np.int64)
    dp_record[0][0][0] = 0
    for i in range(1, n_stream + 1):
        for u in range(min(i * (n_uconfig - 1) + 1, upload_btn + 1)):
            for p in range(min(i * (n_pconfig - 1) + 1, process_btn + 1)):
                for du in range(min(n_uconfig, u + 1)):
                    for dp in range(min(n_pconfig, p + 1)):
                        if dp_record[i - 1, u - du, p - dp] + profit_matrix[du, dp, i - 1] > dp_record[i, u, p]:
                            dp_record[i, u, p] = dp_record[i - 1, u - du, p - dp] + profit_matrix[du, dp, i - 1]
                            choice_record[i, u, p, :] = [du, dp]
    choice = np.zeros((n_stream, 2), dtype=np.int64)
    u_remain, p_remain = upload_btn, process_btn
    for i in range(n_stream, 0, -1):
        choice[i - 1, :] = choice_record[i, u_remain, p_remain]
        du, dp = choice_record[i, u_remain, p_remain]
        u_remain -= du
        p_remain -= dp
    return choice


def simulation(avg_process_tpt, avg_upload_tpt, optimal, use_dp, size, n_stream):
    cfg = Config.fromfile('configs/custom/ssd.py')
    with open('datasets.json') as f:
        datasets = json.load(f)
    with open(f'configs/cache/map_distill_filter_{size}.pkl', 'rb') as f:
        mmap = pickle.load(f)[:, :, :n_stream, :]
    mmap_total = mmap[:, :, :, -1]
    mmap_filter = mmap[:, :, :, :-1]
    n_dconfig, n_pconfig, n_stream, n_epoch = mmap_filter.shape

    if not optimal:
        with open(f'configs/cache/map_observation_{size}.pkl', 'rb') as f:
            mmap_observation = pickle.load(f)[:, :n_stream, :]
    streams = []
    for i, stream in enumerate(datasets):
        if i >= n_stream:
            break
        streams.append(stream)
    if n_stream > len(datasets):
        print("Mismatch between dataset configuration file and dump file!")
        sys.exit(1)
    batch_size = list(datasets.values())[0]['size'] // n_epoch
    if batch_size != size:
        print("Mismatch between dataset configuration file and dump file!")
        sys.exit(1)
    process_btn, upload_btn = avg_process_tpt * n_stream, avg_upload_tpt * n_stream
    baseline_map_total = mmap_total[avg_upload_tpt, avg_process_tpt - 1]
    aca_map_total = np.zeros(baseline_map_total.shape)
    global results
    results = [[] for _ in range(n_stream)]

    # Since epoch 0 have no distillation yet, start with even allocation
    choices = np.zeros((n_epoch, n_stream, 2), dtype=np.int32)
    choices[0, :, :] = [avg_upload_tpt, avg_process_tpt]

    previouses = []
    for stream in range(n_stream):
        name = streams[stream]
        with open(f'snapshot/result/{name}_0-{batch_size}/000000.pkl', 'rb') as f:
            previous = pickle.load(f)
        previouses.append(previous)

    for epoch in range(n_epoch):
        print(f'Simulating epoch {epoch} ...')
        # collect result based on previous choice
        for stream in range(n_stream):
            name = streams[stream]
            upload_cfg, filter_cfg = choices[epoch, stream]
            stride = ORIGINAL_FRAMERATE // filter_cfg
            for i in range(epoch * batch_size, epoch * batch_size + batch_size):
                if i % stride == 0:
                    with open(f'snapshot/result/{name}_{upload_cfg}-{batch_size}/{i:06d}.pkl', 'rb') as f:
                        o = pickle.load(f)
                        results[stream].append(o)
                        previouses[stream] = o
                else:
                    results[stream].append(copy.deepcopy(previouses[stream]))
        # decide filter choice for next epoch
        if epoch + 1 < n_epoch:
            if optimal:
                # jointly allocation bandwidth and computation with optimal knowledge
                mmap_observation_epoch = mmap_filter[:, :, :, epoch + 1]
                choices[epoch + 1, :, :] = allocate2d(upload_btn, process_btn - n_stream, mmap_observation_epoch)
                choices[epoch + 1, :] = current_choice
            else:
                mmap_distill_epoch = mmap_observation[:, :, epoch + 1]
                mmap_filter_epoch = np.transpose(mmap_filter[choices[epoch, :, 0], :, np.arange(n_stream), epoch])
                if use_dp:
                    # first decide bandwidth allocation
                    if epoch == 0:
                        choices[epoch + 1, :, 0] = avg_upload_tpt
                    else:
                        choices[epoch + 1, :, 0] = allocate(upload_btn, mmap_distill_epoch)
                    # then decide computation allocation
                    choices[epoch + 1, :, 1] = allocate(process_btn - n_stream, mmap_filter_epoch)
                    choices[epoch + 1, :] = current_choice
                else:
                    # first decide bandwidth allocation
                    if epoch == 0:
                        choices[epoch + 1, :, 0] = avg_upload_tpt
                    else:
                        current_choice = copy.deepcopy(choices[epoch, :, 0])
                        for _ in range(n_stream):
                            gradient = np.zeros(n_stream, dtype=np.double)
                            for stream in range(n_stream):
                                name = streams[stream]
                                if current_choice[stream] == 0:
                                    gradient[stream] = mmap_distill_epoch[current_choice[stream], stream]
                                else:
                                    gradient[stream] = mmap_distill_epoch[current_choice[stream], stream] - mmap_distill_epoch[current_choice[stream] - 1, stream]
                            index = np.argsort(gradient)
                            thief, victim = n_stream - 1, 0
                            while thief >= 0 and current_choice[index[thief]] == n_dconfig - 1:
                                thief -= 1
                            while victim < n_dconfig and current_choice[index[victim]] == 0:
                                victim += 1
                            if thief > victim:
                                current_choice[index[thief]] += 1
                                current_choice[index[victim]] -= 1
                            else:
                                break
                        choices[epoch + 1, :, 0] = current_choice
                    # then decide computation allocation
                    current_choice = copy.deepcopy(choices[epoch, :, 1])
                    for _ in range(n_stream):
                        gradient = np.zeros(n_stream, dtype=np.double)
                        for stream in range(n_stream):
                            if current_choice[stream] == 1:
                                gradient[stream] = mmap_filter_epoch[current_choice[stream] - 1, stream]
                            else:
                                gradient[stream] = mmap_filter_epoch[current_choice[stream] - 1, stream] - mmap_filter_epoch[current_choice[stream] - 2, stream]
                        index = np.argsort(gradient)
                        thief, victim = n_stream - 1, 0
                        while thief >= 0 and current_choice[index[thief]] == n_pconfig:
                            thief -= 1
                        while victim < n_stream and current_choice[index[victim]] == 1:
                            victim += 1
                        if thief > victim:
                            current_choice[index[thief]] += 1
                            current_choice[index[victim]] -= 1
                        else:
                            break
                    choices[epoch + 1, :, 1] = current_choice

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
    parser.add_argument('--upload', '-u', type=int, default=3, help='average uplink throughput for each stream')
    parser.add_argument('--process', '-p', type=int, default=3, help='average processing throughput for each stream')
    parser.add_argument('--optimal', '-o', type=ast.literal_eval, default=False, help='use optimal knowledge')
    parser.add_argument('--size', '-s', type=int, default=600, help='epoch size')
    parser.add_argument('--aggresive', '-a', type=ast.literal_eval, default=True, help='use DP')
    parser.add_argument('--stream', '-n', type=int, default=12, help='number of streams')
    args = parser.parse_args()
    baseline_map_total, aca_map_total = simulation(args.upload, args.process, args.optimal, args.aggresive, args.size, args.stream)
    print('baseline')
    for v in baseline_map_total:
        print(v)
    print('aca')
    for v in aca_map_total:
        print(v)
