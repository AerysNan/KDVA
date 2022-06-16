from os.path import join as opj

from util.model_test import test

import argparse
import config
import os


def pip_val(dataset, n_window, cfg='configs/custom/ssd_amlt.py', **_):
    o_anno_path = opj(os.environ['AMLT_OUTPUT_DIR'], config.O_ANNO_DIR)
    o_model_path = opj(os.environ['AMLT_OUTPUT_DIR'], config.O_MODEL_DIR)
    o_val_path = opj(os.environ['AMLT_OUTPUT_DIR'], config.O_VAL_DIR)

    os.makedirs(o_anno_path, exist_ok=True)
    os.makedirs(o_model_path, exist_ok=True)
    os.makedirs(o_val_path, exist_ok=True)

    i_data_path = opj(os.environ['AMLT_DATA_DIR'], config.I_DATA_DIR)

    for epoch in range(1, n_window):
        print(f'Validate epoch {epoch}')
        test(
            config=cfg,
            checkpoint=opj(o_model_path, f'{dataset}.{epoch}'),
            out=opj(o_val_path, f'{dataset}.{epoch}.pkl'),
            anno_file=opj(o_anno_path, config.VAL_DIR, f'{dataset}.{epoch - 1}'),
            img_prefix=i_data_path
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validation pipeline')
    parser.add_argument('--dataset', '-d', help='dataset name', type=str, required=True)
    parser.add_argument('--n-window', '-n', help='number of training windows', type=int, default=30)
    parser.add_argument('--cfg', '-c', help='test configuration', type=str, default='configs/custom/ssd_amlt.py')
    args = parser.parse_args()
    pip_val(**args.__dict__)
