import sys
import json
import copy
import numpy as np
import pickle
import argparse

from mmcv import Config
from mmdet.datasets import build_dataset
from multiprocessing import Pool


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


parser = argparse.ArgumentParser(description='Fake replay')
parser.add_argument('--throughput', '-t', type=int, default=3, help='average uplink throughput for each stream')
args = parser.parse_args()

average_tpt = args.throughput

with open('datasets.json') as f:
    datasets = json.load(f)

with open('tools/mmap_distill.pkl', 'rb') as f:
    mmap = pickle.load(f)
mmap_total = mmap[:, -1, :]
mmap_distill = mmap[:, :-1, :]
n_config, n_epoch, n_stream = mmap_distill.shape
with open('tools/mmap_observation.pkl', 'rb') as f:
    mmap_observation = pickle.load(f)

if n_stream != len(datasets):
    print("Mismatch between dataset configuration file and dump file!")
    sys.exit(1)
batch_size = list(datasets.values())[0]['size'] // n_epoch
bottleneck = average_tpt * n_stream
baseline_map_distill, baseline_map_total = mmap_distill[average_tpt, :, :], mmap_total[average_tpt]
aca_map_distill, aca_map_total = np.zeros(baseline_map_distill.shape), np.zeros(baseline_map_total.shape)
results = [[] for _ in range(n_stream)]

# Since epoch 0 have no distillation yet, start with even allocation
choices = np.zeros((n_epoch, n_stream), dtype=np.int64)
choices[0, :] = average_tpt

for epoch in range(n_epoch):
    print(f'Simulating epoch {epoch} ...')
    # collect result based on previous choice
    for name in datasets:
        stream = datasets[name]['id']
        for i in range(epoch * batch_size, epoch * batch_size + batch_size):
            with open(f'snapshot/result/{name}_{choices[epoch, stream]}/{i:06d}.pkl', 'rb') as f:
                result = pickle.load(f)
            results[stream].append(result)
        aca_map_distill[epoch, stream] = mmap_distill[choices[epoch, stream], epoch, stream]
    # decide distillation choice for next epoch
    if epoch + 1 < n_epoch:
        # use observed outcome online
        mmap_observation_epoch = mmap_observation[:, epoch, :]  # n_config * n_stream
        current_choice = copy.deepcopy(choices[epoch, :])
        for _ in range(n_stream // 2):
            gradient = np.zeros(n_stream, dtype=np.double)
            for name in datasets:
                stream = datasets[name]['id']
                if current_choice[stream] == 0:
                    gradient[stream] = 1
                else:
                    gradient[stream] = mmap_observation_epoch[current_choice[stream], stream] - mmap_observation_epoch[current_choice[stream] - 1, stream]
            index = np.argsort(gradient)
            thief, victim = n_stream - 1, 0
            while thief >= 0 and current_choice[index[thief]] == n_config - 1:
                thief -= 1
            while victim < n_config and current_choice[index[victim]] == 0:
                victim += 1
            if thief > victim:
                current_choice[index[thief]] += 1
                current_choice[index[victim]] -= 1
                print(f'stream {index[thief]} steal from stream {index[victim]}')
            # overwrite outcome that requried upsampling
            # for name in datasets:
            #     stream = datasets[name]['id']
            #     for config in range(max(1, choices[epoch, stream]) + 1, n_config):
            #         mmap_observation_epoch[config, stream] = 2 * mmap_observation_epoch[config - 1, stream] - mmap_observation_epoch[config - 2, stream]
            # use groud truth outcome
            # mmap_observation_epoch = mmap_distill[:, epoch + 1, :]

            # start DP
            # current_choice = allocate(bottleneck, mmap_observation_epoch)
        choices[epoch + 1, :] = current_choice


def eval(cfg, stream):
    # define evaluation function for multiprocessing
    print(f'Evaluting stream {stream} ...')
    dataset = build_dataset(cfg.data.test)
    d = {
        'id': stream,
        'mAP': dataset.evaluate(results[stream], metric='bbox')['bbox_mAP'],
    }
    print(f'Evaluting stream {stream} finished!')
    return d


print('Simulation ended, starting evaluation ...')
pool, output = Pool(processes=n_stream // 3), []

for name in datasets:
    stream = datasets[name]['id']
    cfg = Config.fromfile('configs/custom/ssd.py')
    cfg.data.test.ann_file = f'data/annotations/{name}.gt.json'
    cfg.data.test.img_prefix = ''
    output.append(pool.apply_async(eval, (cfg, stream,)))
pool.close()
pool.join()
for i in range(n_stream):
    result = output[i].get()
    aca_map_total[result['id']] = result['mAP']

print(f'stream level optimized mAP: {aca_map_total}')
print(f'baseline mAP = {baseline_map_total.sum()}, optimized mAP = {aca_map_total.sum()}')
