from shutil import copyfile
from os.path import join as opj
from util.model_train import train

import argparse
import config
import os


def pip_aggregate(datasets, n_window, aggregation=1, agg_name=None, cfg='configs/custom/ssd_amlt.py', backbone_cfg='configs/custom/ssd_amlt.py', ** _):
    o_anno_path = opj(os.environ['AMLT_OUTPUT_DIR'], config.O_ANNO_DIR)
    o_model_path = opj(os.environ['AMLT_OUTPUT_DIR'], config.O_MODEL_DIR)
    o_log_path = opj(os.environ['AMLT_OUTPUT_DIR'], config.O_LOG_DIR)

    os.makedirs(o_anno_path, exist_ok=True)
    os.makedirs(o_model_path, exist_ok=True)
    os.makedirs(o_log_path, exist_ok=True)

    i_data_path = opj(os.environ['AMLT_DATA_DIR'], config.I_DATA_DIR)
    i_model_path = opj(os.environ['AMLT_DATA_DIR'], config.I_MODEL_DIR)

    datasets = datasets if type(datasets) == list else [datasets]
    for dataset in datasets:
        copyfile(opj(i_model_path, config.STUDENT_MODEL), opj(o_model_path, f'{dataset}.0'))
    copyfile(opj(i_model_path, config.STUDENT_MODEL), opj(o_model_path, f'{agg_name}.0'))
    for epoch in range(1, n_window):
        if epoch % aggregation == 0:
            print(f'Train aggregated backbone on epoch {epoch}')
            train(
                config=backbone_cfg,
                work_dir=o_log_path,
                train_anno_file=opj(o_anno_path, config.TRAIN_DIR, f'{agg_name}.{epoch // aggregation - 1}'),
                train_img_prefix=i_data_path,
                seed=0,
                deterministic=True,
                load_from=opj(i_model_path, config.STUDENT_MODEL)
            )
            copyfile(opj(o_log_path, 'latest.pth'), opj(o_model_path, f'{agg_name}.{epoch // aggregation}'))
        print(f'Train epoch {epoch}')
        for dataset in datasets:
            train(
                config=cfg,
                work_dir=o_log_path,
                train_anno_file=opj(o_anno_path, config.TRAIN_DIR, f'{dataset}.{epoch - 1}'),
                train_img_prefix=i_data_path,
                seed=0,
                deterministic=True,
                load_from=opj(o_model_path, f'{agg_name}.{epoch // aggregation}')
            )
            copyfile(opj(o_log_path, 'latest.pth'), opj(o_model_path, f'{dataset}.{epoch}'))
    print('Train finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aggregated training pipeline')
    parser.add_argument('--datasets', '-d', nargs='+', help='datasets in one aggregation', required=True)
    parser.add_argument('--n-window', '-n', help='number of training windows', type=int, default=30)
    parser.add_argument('--cfg', '-c', help='train head configuration', type=str, default='configs/custom/ssd_amlt.py')
    parser.add_argument('--backbone-cfg', '-bc', help='train backbone configuration', type=str, default='configs/custom/ssd_amlt.py')
    parser.add_argument('--agg-name', '-an', help='aggregated dataset name', type=str, default=None)
    parser.add_argument('--aggregation', '-a', help='the number of intervals for temporal aggregation', type=int, default=1)
    args = parser.parse_args()
    pip_aggregate(**args.__dict__)
