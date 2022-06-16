from os.path import join as opj

from util.data_sample import downsample_dataset
from util.data_split import split_dataset
from util.data_merge import merge_traces

import argparse
import config
import json
import os


def pip_split(datasets, n_window, train_rate=1, val_rate=0.1, aggregation=None, agg_name=None, **_):
    with open('cfg_data.json') as f:
        all_datasets = json.load(f)

    o_anno_path = opj(os.environ['AMLT_OUTPUT_DIR'], config.O_ANNO_DIR)

    os.makedirs(o_anno_path, exist_ok=True)

    print('Start generating split dataset...')
    datasets = datasets if type(datasets) == list else [datasets]
    for dataset in datasets:
        window_length = all_datasets[dataset]['size'] // n_window
        split_dataset(
            input_file=opj(o_anno_path, f'{dataset}.json'),
            output_dir=o_anno_path,
            output_name=dataset,
            size=window_length,
            train_rate=f'{train_rate}/{all_datasets[dataset]["fps"]}',
            val_size=int(window_length * val_rate),
        )
    print('Split finished!')
    if aggregation is None:
        return
    print('Start aggregating dataset...')
    agg_name = config.AGG_NAME if agg_name is None else agg_name
    for epoch in range(0, n_window, aggregation):
        agg_datasets = []
        for dataset in datasets:
            for i in range(aggregation):
                agg_datasets.append(opj(o_anno_path, config.TRAIN_DIR, f'{dataset}.{i + epoch}'))
        merge_traces(
            input_file=agg_datasets,
            output_file=opj(o_anno_path, config.TRAIN_DIR, f'{agg_name}.{epoch // aggregation}')
        )
    print('Aggregation finished!')
    print('Start downsampling dataset...')
    for epoch in range(0, n_window, aggregation):
        downsample_dataset(
            input_file=opj(o_anno_path, config.TRAIN_DIR, f'{agg_name}.{epoch // aggregation}'),
            output_file=opj(o_anno_path, config.TRAIN_DIR, f'{agg_name}.{epoch // aggregation}'),
            rate=len(datasets) * aggregation
        )
    print('Downsampling finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset splitting pipeline')
    parser.add_argument('--datasets', '-d', nargs='+', help='datasets in one aggregation', required=True)
    parser.add_argument('--n-window', '-n', help='number of training windows', type=int, default=30)
    parser.add_argument('--train-rate', '-tr', help='training rate', type=int, default=1)
    parser.add_argument('--val-rate', '-vr', help='validation rate', type=float, default=0.1)
    parser.add_argument('--agg-name', '-an', help='aggregated dataset name', type=str, default=None)
    parser.add_argument('--aggregation', '-a', help='the number of intervals for temporal aggregation', type=int, default=None)
    args = parser.parse_args()
    pip_split(**args.__dict__)
