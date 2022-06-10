from shutil import copyfile
from os.path import join as opj
from tools.model_test import test
from tools.model_train import train
from tools.split_dataset import split_dataset
from tools.anno_from_result import anno_from_result
import argparse
import json
import ast
import os

FPS = 30

I_DATA_DIR = 'data'
I_ANNO_DIR = 'data/annotations'
I_MODEL_DIR = 'models'

O_MODEL_DIR = 'models'
O_RESULT_DIR = 'results'
O_LOG_DIR = 'log'


def train_test_na(dataset, n_window, train_rate=1, val_rate=0.1, threshold=None, cfg='base', eval=False, work_dir=None, **_):
    if work_dir is None:
        work_dir = f'{dataset}_{train_rate}'

    with open('cfg_data.json') as f:
        datasets = json.load(f)
    window_length = datasets[dataset]["size"] // n_window

    o_model_path = opj(os.environ['AMLT_OUTPUT_DIR'], work_dir, O_MODEL_DIR)
    o_result_path = opj(os.environ['AMLT_OUTPUT_DIR'], work_dir, O_RESULT_DIR)
    o_log_path = opj(os.environ['AMLT_OUTPUT_DIR'], work_dir, O_LOG_DIR)

    os.makedirs(o_model_path, exist_ok=True)
    os.makedirs(o_result_path, exist_ok=True)
    os.makedirs(o_log_path, exist_ok=True)

    i_model_student = opj(os.environ['AMLT_DATA_DIR'], I_MODEL_DIR, 'ssd.pth')
    i_model_teacher = opj(os.environ['AMLT_DATA_DIR'], I_MODEL_DIR, 'r101.pth')
    copyfile(i_model_student, opj(o_model_path, '0.pth'))
    if threshold is not None:
        print('Annotation file updated, start generating...')
        test(
            config=f'configs/custom/rcnn.py',
            checkpoint=i_model_teacher,
            out=opj(o_log_path, f'{dataset}.golden.pkl'),
            dataset=dataset,
            root=os.environ['AMLT_DATA_DIR']
        )
        anno_from_result(
            root=os.environ['AMLT_DATA_DIR'],
            path=opj(o_log_path, f'{dataset}.golden.pkl'),
            dataset=dataset,
            threshold=threshold
        )
        print('Pesudo labels generated!')
    if not os.path.exists(opj(os.environ['AMLT_DATA_DIR'], I_ANNO_DIR, f'{dataset}_{train_rate}_train_0.golden.json')):
        print('Windowed dataset not found, start splitting...')
        split_dataset(
            path=os.environ['AMLT_DATA_DIR'],
            dataset=dataset,
            size=window_length,
            train_rate=f'{train_rate}/{FPS}',
            val_size=int(window_length * val_rate),
            postfix=train_rate
        )
        print('Split finished!')
    print('Test epoch 0')
    test(
        config=f'configs/custom/ssd_{cfg}.py',
        checkpoint=opj(o_model_path, '0.pth'),
        out=opj(o_result_path, '00.pkl'),
        dataset=f'{dataset}_test_0',
        root=os.environ['AMLT_DATA_DIR']
    )
    for epoch in range(1, n_window):
        print(f'Train epoch {epoch}')
        train(
            config=f'configs/custom/ssd_{cfg}.py',
            work_dir=o_log_path,
            root=os.environ['AMLT_DATA_DIR'],
            train_dataset=f'{dataset}_{train_rate}_train_{epoch - 1}',
            seed=0,
            deterministic=True,
            load_from=i_model_student
        )
        copyfile(opj(o_log_path, 'latest.pth'), opj(o_model_path, f'{epoch}.pth'))
        print(f'Test epoch {epoch}')
        test(
            config=f'configs/custom/ssd_{cfg}.py',
            checkpoint=opj(o_model_path, f'{epoch}.pth'),
            out=opj(o_result_path, f'{epoch:02d}.pkl'),
            dataset=f'{dataset}_test_{epoch}',
            root=os.environ['AMLT_DATA_DIR']
        )

    if not eval:
        return
    print('start evaluation...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train and test in a non-accumulative way")
    parser.add_argument("--dataset", "-d", help="dataset name", type=str, required=True)
    parser.add_argument("--train-rate", "-r", help="training rate", type=int, required=True)
    parser.add_argument("--n-window", "-n", help="number of training windows", type=int, default=30)
    parser.add_argument("--threshold", "-t", help="annotation threshold", type=float, default=None)
    parser.add_argument("--cfg", "-c", help="train and test configuration", type=str, default='base')
    parser.add_argument("--eval", "-e", help="evalutaion or not", type=ast.literal_eval, default=False)
    parser.add_argument("--work-dir", "-w", help="working directory", type=str, default=None)
    args = parser.parse_args()
    train_test_na(**args.__dict__)
