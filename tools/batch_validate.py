import os
import argparse
from model_test import test


def batch_validate(path, dataset, stream_l, stream_r, retrain_l, retrain_r, postfix, vals, epoch, **_):
    for stream in range(stream_l, stream_r + 1):
        for retrain in range(retrain_l, retrain_r + 1):
            for val in vals:
                os.makedirs(f'{path}/snapshot/result/{dataset}_{stream}_{retrain}_{postfix}v{val}', exist_ok=True)
                for e in range(1, epoch):
                    test('configs/custom/ssd_base.py', f'{path}/snapshot/models/{dataset}_{stream}_{retrain}_{postfix}/{e}.pth',
                         datapath=path,
                         dataset=f'{dataset}_{stream}_{val}_val_{e-1}',
                         out=f'{path}/snapshot/result/{dataset}_{stream}_{retrain}_{postfix}v{val}/{e:02d}.pkl'
                         )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch validation')
    parser.add_argument('--path', '-p', type=str, required=True)
    parser.add_argument('--dataset', '-d', type=str, default='detrac')
    parser.add_argument('--stream-l', type=int, default=1)
    parser.add_argument('--stream-r', type=int, default=6)
    parser.add_argument('--retrain-l', type=int, default=0)
    parser.add_argument('--retrain-r', type=int, default=6)
    parser.add_argument('--postfix', type=str, default='e40')
    parser.add_argument('--vals', nargs="+", default=[f'{pos}{size}' for pos in ['even'] for size in range(20, 120, 20)])
    parser.add_argument('--epoch', type=int, default=12)
    args = parser.parse_args()
    batch_validate(**args.__dict__)
