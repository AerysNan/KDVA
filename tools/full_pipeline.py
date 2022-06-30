import os
from pip_prep import pip_preprocess
from pip_split import pip_split
from pip_train import pip_train
from pip_agg import pip_aggregate
from pip_test import pip_test
from pip_val import pip_val
from pip_eval import pip_eval

import argparse
import datetime


def broadcast_wrapper(f):
    def broadcast_f(datasets, **kwargs):
        for dataset in datasets:
            f(dataset=dataset, **kwargs)
    return broadcast_f


def full_pipeline(start=0, **kwargs):
    PIPELINE = [
        ('preprocess', broadcast_wrapper(pip_preprocess)),
        ('split', pip_split),
        # ('train', broadcast_wrapper(pip_train) if kwargs['aggregation'] is None else pip_aggregate),
        ('test', broadcast_wrapper(pip_test)),
        ('val', broadcast_wrapper(pip_val)),
        ('evaluation', broadcast_wrapper(pip_eval)),
    ]
    for i in range(start, len(PIPELINE)):
        name, step = PIPELINE[i]
        print(f'Start step {i}: {name}, current time: {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
        step(**kwargs)
        print(f'Finish step {i}: {name}, current time: {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full pipeline')
    # general args
    parser.add_argument('--datasets', '-d', nargs='+', help='datasets in one aggregation', required=True)
    parser.add_argument('--start', '-s', help='start from which step', type=int, default=0)
    # preprocess args
    parser.add_argument('--anno-threshold', '-t', help='annotation threshold', type=float, default=0.5)
    parser.add_argument('--anno-template', '-at', help='annotation template', type=str, default=None)
    parser.add_argument('--base-template', '-bt', help='base template', type=str, default=None)
    parser.add_argument('--result-template', '-rt', help='result template', type=str, default=None)
    # split args
    parser.add_argument('--n-window', '-n', help='number of training windows', type=int, default=30)
    parser.add_argument('--train-rate', '-tr', help='training rate', type=int, default=1)
    parser.add_argument('--val-rate', '-vr', help='validation rate', type=float, default=0.1)
    parser.add_argument('--agg-name', '-an', help='aggregated dataset name', type=str, default=None)
    parser.add_argument('--aggregation', '-a', help='the number of intervals for temporal aggregation', type=int, default=None)
    # train test val args
    parser.add_argument('--cfg', '-c', help='train test val configuration', type=str, default='configs/custom/ssd_amlt.py')
    parser.add_argument('--backbone-cfg', '-bc', help='train backbone configuration', type=str, default='configs/custom/ssd_amlt.py')
    # eval args
    parser.add_argument('--eval-template', '-et', help='evaluation file template', type=str, required=True)
    parser.add_argument('--n-process', '-np', help='number of concurrent process', type=int, default=4)
    parser.add_argument('--framerates', nargs='+', default=[], help='inference framerate levels used for evaluation')
    args = parser.parse_args()
    with open(os.path.join(os.environ['AMLT_OUTPUT_DIR'], 'args'), 'w') as f:
        f.write(str(vars(args)))
    full_pipeline(**args.__dict__)
