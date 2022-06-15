from shutil import copyfile
from os.path import join as opj
from tools.model_test import test
from tools.model_train import train
from tools.split_dataset import split_dataset
from tools.anno_from_imgs import anno_from_imgs
from tools.anno_from_result import anno_from_result

import argparse
import json
import ast
import os

FPS = 30

I_DATA_DIR = 'data'
I_ANNO_DIR = 'annos'
I_MODEL_DIR = 'models'

O_MODEL_DIR = 'models'
O_ANNO_DIR = 'annos'
O_RESULT_DIR = 'results'
O_LOG_DIR = 'log'

TRAIN_DIR = 'train'
VAL_DIR = 'val'
TEST_DIR = 'test'

STUDENT_MODEL = 'ssd.pth'
TEACHER_MODEL = 'r101.pth'


def train_test_na(dataset, n_window, train_rate=1, val_rate=0.1, anno_threshold=0.5, base_file=None, result_file=None, anno_file=None, cfg='configs/custom/ssd_amlt.py', eval=False, no_test=True, **_):
    with open('cfg_data.json') as f:
        datasets = json.load(f)
    window_length = datasets[dataset]["size"] // n_window

    o_anno_path = opj(os.environ['AMLT_OUTPUT_DIR'], O_ANNO_DIR)
    o_model_path = opj(os.environ['AMLT_OUTPUT_DIR'], O_MODEL_DIR)
    o_result_path = opj(os.environ['AMLT_OUTPUT_DIR'], O_RESULT_DIR)
    o_log_path = opj(os.environ['AMLT_OUTPUT_DIR'], O_LOG_DIR)

    os.makedirs(o_anno_path, exist_ok=True)
    os.makedirs(o_model_path, exist_ok=True)
    os.makedirs(o_result_path, exist_ok=True)
    os.makedirs(o_log_path, exist_ok=True)

    i_data_path = opj(os.environ['AMLT_DATA_DIR'], I_DATA_DIR)
    i_anno_path = opj(os.environ['AMLT_DATA_DIR'], I_ANNO_DIR)
    i_model_path = opj(os.environ['AMLT_DATA_DIR'], I_MODEL_DIR)

    if anno_file is not None:
        print('Annotation file provided, skip generation!')
        copyfile(opj(i_anno_path, anno_file), opj(o_anno_path, f'{dataset}.json'))
    else:
        if base_file is not None:
            print('Dataset description file provided, skip generation!')
            copyfile(opj(i_anno_path, base_file), opj(o_anno_path, f'{dataset}.b.json'))
        else:
            print('Start generating dataset description file...')
            anno_from_imgs(
                img_path=opj(i_data_path, dataset),
                classes='classes.dat',
                out=opj(o_anno_path, f'{dataset}.b.json'),
                prefix=dataset
            )
            print('Dataset description file generated!')
        if result_file is not None:
            print('Expert model inference result provided, skip inference!')
            copyfile(opj(i_anno_path, result_file), opj(o_log_path, f'{dataset}.r.pkl'))
        else:
            print('Start inferencing with expert model...')
            test(
                config='configs/custom/rcnn_amlt.py',
                checkpoint=opj(i_model_path, TEACHER_MODEL),
                out=opj(o_log_path, f'{dataset}.r.pkl'),
                anno_file=opj(o_anno_path, f'{dataset}.b.json'),
                img_prefix=i_data_path
            )
            print('Inference finished!')
        print('Start generating annotation file...')
        anno_from_result(
            base_file=opj(i_anno_path, f'{dataset}.b.json'),
            result_file=opj(o_log_path, f'{dataset}.r.pkl'),
            output_file=opj(o_anno_path, f'{dataset}.json'),
            threshold=anno_threshold
        )
        print('Annotation file generated!')

    print('Start generating split dataset...')
    split_dataset(
        input_file=opj(o_anno_path, f'{dataset}.json'),
        output_dir=o_anno_path,
        output_name=dataset,
        size=window_length,
        train_rate=f'{train_rate}/{FPS}',
        val_size=int(window_length * val_rate),
    )
    print('Split finished!')

    copyfile(opj(i_model_path, STUDENT_MODEL), opj(o_model_path, '0.pth'))
    if not no_test:
        print('Test epoch 0')
        test(
            config=cfg,
            checkpoint=opj(o_model_path, '0.pth'),
            out=opj(o_result_path, '00.pkl'),
            anno_file=opj(o_anno_path, TEST_DIR, f'{dataset}.{0}'),
            img_prefix=i_data_path
        )
    for epoch in range(1, n_window):
        print(f'Train epoch {epoch}')
        train(
            config=cfg,
            work_dir=o_log_path,
            train_anno_file=opj(o_anno_path, TRAIN_DIR, f'{dataset}.{epoch - 1}'),
            train_img_prefix=i_data_path,
            seed=0,
            deterministic=True,
            load_from=opj(i_model_path, STUDENT_MODEL)
        )
        copyfile(opj(o_log_path, 'latest.pth'), opj(o_model_path, f'{epoch}.pth'))
        if not no_test:
            print(f'Test epoch {epoch}')
            test(
                config=cfg,
                checkpoint=opj(o_model_path, f'{epoch}.pth'),
                out=opj(o_result_path, f'{epoch:02d}.pkl'),
                anno_file=opj(o_anno_path, TEST_DIR, f'{dataset}.{epoch}'),
                img_prefix=i_data_path
            )
    if not eval:
        return
    print('start evaluation...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train and test in a non-accumulative way")
    parser.add_argument("--dataset", "-d", help="dataset name", type=str, required=True)
    parser.add_argument("--n-window", "-n", help="number of training windows", type=int, default=30)
    parser.add_argument("--train-rate", "-tr", help="training rate", type=int, default=1)
    parser.add_argument("--val-rate", "-vr", help="validation rate", type=float, default=0.1)
    parser.add_argument("--anno-threshold", "-t", help="annotation threshold", type=float, default=0.5)
    parser.add_argument("--anno-file", "-af", help="annotation file", type=str, default=None)
    parser.add_argument("--base-file", "-bf", help="base file", type=str, default=None)
    parser.add_argument("--base-file", "-rf", help="result file", type=str, default=None)
    parser.add_argument("--cfg", "-c", help="train and test configuration", type=str, default='configs/custom/ssd_amlt.py')
    parser.add_argument("--eval", "-e", help="evalutaion or not", type=ast.literal_eval, default=False)
    parser.add_argument("--no-test", "-nt", help="no test", type=ast.literal_eval, default=True)
    args = parser.parse_args()
    train_test_na(**args.__dict__)
