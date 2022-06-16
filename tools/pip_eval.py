from multiprocessing import Pool
from os.path import join as opj

from util.result_evaluate import evaluate_from_file

import argparse
import pickle
import config
import json
import os


def pip_eval(dataset, n_window, eval_file=None, n_process=4, framerates=[], **kwargs):
    if 'eval_template' in kwargs and type(kwargs['eval_template']) == str:
        eval_file = kwargs['eval_template'].format(dataset)

    with open('cfg_data.json') as f:
        datasets = json.load(f)
    original_framerate = datasets[dataset]['fps']
    if len(framerates) == 0:
        framerates.append(original_framerate)
    elif type(framerates[0]) == str:
        framerates = list(map(int, framerates))

    o_anno_path = opj(os.environ['AMLT_OUTPUT_DIR'], config.O_ANNO_DIR)
    o_result_path = opj(os.environ['AMLT_OUTPUT_DIR'], config.O_RESULT_DIR)
    o_val_path = opj(os.environ['AMLT_OUTPUT_DIR'], config.O_VAL_DIR)

    os.makedirs(o_anno_path, exist_ok=True)
    os.makedirs(o_result_path, exist_ok=True)

    i_data_path = opj(os.environ['AMLT_DATA_DIR'], config.I_DATA_DIR)

    d = {}
    pool_output, p = {}, Pool(processes=n_process)
    for framerate in framerates:
        for epoch in range(n_window):
            pool_output[('test', epoch, framerate)] = p.apply_async(evaluate_from_file, (
                opj(o_result_path, f'{dataset}.{epoch}.pkl'),
                opj(o_anno_path, config.TEST_DIR, f'{dataset}.{epoch}'),
                i_data_path,
                (framerate, original_framerate),
                'configs/custom/ssd_amlt.py'
            ))
        for epoch in range(1, n_window):
            pool_output[('val', epoch, framerate)] = p.apply_async(evaluate_from_file, (
                opj(o_val_path, f'{dataset}.{epoch}.pkl'),
                opj(o_anno_path, config.VAL_DIR, f'{dataset}.{epoch - 1}'),
                i_data_path,
                (framerate, original_framerate),
                'configs/custom/ssd_amlt.py',
            ))
    p.close()
    p.join()
    for framerate in framerates:
        for epoch in range(n_window):
            result = pool_output[('test', epoch, framerate)].get(600)
            d[('test', epoch, framerate)] = result
        for epoch in range(1, n_window):
            result = pool_output[('val', epoch, framerate)].get(600)
            d[('val', epoch, framerate)] = result
    with open(opj(os.environ['AMLT_OUTPUT_DIR'], eval_file), 'wb') as framerate:
        pickle.dump(d, framerate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation pipeline')
    parser.add_argument('--dataset', '-d', help='dataset name', type=str, required=True)
    parser.add_argument('--eval-file', '-ef', help='evaluation file name', type=str, required=True)
    parser.add_argument('--n-window', '-n', help='number of training windows', type=int, default=30)
    parser.add_argument('--n-process', '-p', help='number of concurrent process', type=int, default=4)
    parser.add_argument('--framerates', nargs='+', default=[], help='inference framerate levels used for evaluation')
    args = parser.parse_args()
    pip_eval(**args.__dict__)
