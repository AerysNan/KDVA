import os
from shutil import copyfile
import json
import copy
import numpy as np
import argparse
import pickle

from evaluate_from_file import evaluate_from_file
from train_new import train
from torch.multiprocessing import Pool
from test_new import test
from merge_result import merge_result


def choice_to_distill(choice):
    return choice * 20


def parallel_test(stream, epoch, gpu, model, out):
    print(f'test stream {stream} at epoch {epoch} with model {model}')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    try:
        test('configs/custom/ssd_all.py', model, f'tmp_{stream}/{out}.pkl', f'{stream}_test_{epoch}')
    except Exception as err:
        print(err)
    print(f'test stream {stream} at epoch {epoch} with model {model} finished')


def parallel_train(stream, retrain, epoch, gpu, postfix):
    print(f'train stream {stream} at epoch {epoch} with retrain {retrain}')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    try:
        train('configs/custom/ssd_all.py', f'tmp_{stream}/{retrain}', f'{stream}_{choice_to_distill(retrain):03d}-{postfix}_train_{epoch}', None, f'tmp_{stream}/latest.pth')
    except Exception as err:
        print(err)
    print(f'train stream {stream} at epoch {epoch} with retrain {retrain} finished')


def fake_distill(average_tpt, n_epoch, n_config, postfix):
    with open('datasets.json') as f:
        datasets = json.load(f)
    n_stream, streams = len(datasets), []
    for i, stream in enumerate(datasets):
        streams.append(stream)
        os.makedirs(f'tmp_{stream}/0', exist_ok=True)
        copyfile('checkpoints/ssd.pth', f'tmp_{stream}/latest.pth')
        parallel_test(stream, 0, 0, f'tmp_{stream}/latest.pth', '00')
    choices = np.zeros((n_epoch, n_stream), dtype=np.int64)
    choices[0, :] = average_tpt

    mmap_observation_epoch = np.zeros((n_config, n_stream), dtype=np.double)

    for epoch in range(n_epoch):
        # decide distillation choice for next epoch
        print('decide distillation choice for next epoch')
        if epoch + 1 < n_epoch:
            if epoch == 0:
                choices[epoch + 1, :] = average_tpt
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
            # explore all retraining configurations
            print('explore all retraining configurations')
            for stream in range(n_stream):
                for retrain in range(1, n_config):
                    parallel_train(streams[stream], retrain, epoch, ((stream) * n_config + retrain) % 2, postfix)
                copyfile(f'tmp_{streams[stream]}/latest.pth', f'tmp_{streams[stream]}/0/latest.pth')
            # test all retraining configurations
            print('test all retraining configurations')
            for stream in range(n_stream):
                for retrain in range(n_config):
                    parallel_test(streams[stream], epoch + 1, (stream * n_config + retrain) % 2, f'tmp_{streams[stream]}/{retrain}/latest.pth', f'tmp/{epoch + 1:02d}-{retrain}')
            # generate mAP observation profile
            print('generate mAP observation profile')
            for stream in range(n_stream):
                for retrain in range(n_config):
                    mmap_observation_epoch[retrain, stream] = evaluate_from_file(f'tmp_{streams[stream]}/tmp/{epoch + 1:02d}-{retrain}.pkl',
                                                                                 f'data/annotations/{streams[stream]}_test_{epoch + 1}.gt.json')['bbox_mAP']
            print(mmap_observation_epoch)
        # update latest model and generate test results
        print('update latest model and generate test results')
        for stream in range(n_stream):
            copyfile(f'tmp_{streams[stream]}/{choices[epoch + 1, stream]}/latest.pth', f'tmp_{streams[stream]}/latest.pth')
            copyfile(f'tmp_{streams[stream]}/tmp/{epoch + 1:02d}-{choices[epoch + 1, stream]}.pkl', f'tmp_{streams[stream]}/{epoch + 1:02d}.pkl')

    print('Simulation ended, starting evaluation ...')
    print(choices)
    for stream in range(n_stream):
        merge_result(f'tmp_{streams[stream]}', f'tmp_{streams[stream]}/merge.pkl')
    pool, output = Pool(processes=4), []
    for stream in range(n_stream):
        name = streams[stream]
        output.append(pool.apply_async(evaluate_from_file, (f'tmp_{name}/merge.pkl', f'data/annotations/{name}.gt.json',)))
    pool.close()
    pool.join()
    final_result = []
    for i in range(n_stream):
        try:
            final_result.append(output[i].get())
        except:
            print('Timeout occurred!')
    return final_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake distill')
    parser.add_argument('--throughput', '-t', type=int, default=3, help='average uplink throughput for each stream')
    args = parser.parse_args()
    result = fake_distill(args.throughput, 20, 6, '500')
    with open('out.pkl', 'wb') as f:
        pickle.dump(result, f)
