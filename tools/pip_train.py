from shutil import copyfile
from os.path import join as opj
from util.model_train import train

import argparse
import config
import os


def pip_train(dataset, n_window, cfg='configs/custom/ssd_amlt.py', **_):
    o_anno_path = opj(os.environ['AMLT_OUTPUT_DIR'], config.O_ANNO_DIR)
    o_model_path = opj(os.environ['AMLT_OUTPUT_DIR'], config.O_MODEL_DIR)
    o_log_path = opj(os.environ['AMLT_OUTPUT_DIR'], config.O_LOG_DIR)

    os.makedirs(o_anno_path, exist_ok=True)
    os.makedirs(o_model_path, exist_ok=True)
    os.makedirs(o_log_path, exist_ok=True)

    i_data_path = opj(os.environ['AMLT_DATA_DIR'], config.I_DATA_DIR)
    i_model_path = opj(os.environ['AMLT_DATA_DIR'], config.I_MODEL_DIR)

    copyfile(opj(i_model_path, config.STUDENT_MODEL), opj(o_model_path, f'{dataset}.0'))
    for epoch in range(1, n_window):
        print(f'Train epoch {epoch}')
        train(
            config=cfg,
            work_dir=o_log_path,
            train_anno_file=opj(o_anno_path, config.TRAIN_DIR, f'{dataset}.{epoch - 1}'),
            train_img_prefix=i_data_path,
            seed=0,
            deterministic=True,
            load_from=opj(i_model_path, config.STUDENT_MODEL)
        )
        copyfile(opj(o_log_path, 'latest.pth'), opj(o_model_path, f'{dataset}.{epoch}'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train pipeline')
    parser.add_argument('--dataset', '-d', help='dataset name', type=str, required=True)
    parser.add_argument('--n-window', '-n', help='number of training windows', type=int, default=30)
    parser.add_argument('--cfg', '-c', help='train configuration', type=str, default='configs/custom/ssd_amlt.py')
    args = parser.parse_args()
    pip_train(**args.__dict__)
