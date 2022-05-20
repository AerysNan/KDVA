import os
import ast
import argparse
from model_test import test

streams = {
    1: '1m12',
    2: '2m12',
    3: '3m34',
    4: '4m34',
}


def batch_validate(path, dataset, stream_l, stream_r, retrain_l, retrain_r, postfix, val, epoch, **_):
    for stream in range(stream_l, stream_r + 1):
        _stream = streams[stream] if postfix == 'agg' else stream
        for retrain in range(retrain_l, retrain_r + 1):
            if val:
                if retrain == 0:
                    os.makedirs(f'{path}/snapshot/result/{dataset}_{_stream}_basev', exist_ok=True)
                    for e in range(1, epoch):
                        test('configs/custom/ssd_base.py', f'{path}/checkpoints/ssd.pth',
                             datapath=path,
                             dataset=f'{dataset}_{stream}_val_{e-1}',
                             out=f'{path}/snapshot/result/{dataset}_{_stream}_basev/{e:02d}.pkl'
                             )
                else:
                    os.makedirs(f'{path}/snapshot/result/{dataset}_{_stream}_{retrain}_{postfix}v', exist_ok=True)
                    for e in range(1, epoch):
                        test('configs/custom/ssd_base.py', f'{path}/snapshot/models/{dataset}_{_stream}_{retrain}_{postfix}/{e}.pth',
                             datapath=path,
                             dataset=f'{dataset}_{stream}_val_{e-1}',
                             out=f'{path}/snapshot/result/{dataset}_{_stream}_{retrain}_{postfix}v/{e:02d}.pkl'
                             )
            else:
                if retrain == 0:
                    os.makedirs(f'{path}/snapshot/result/{dataset}_{_stream}_base', exist_ok=True)
                    for e in range(epoch):
                        test('configs/custom/ssd_base.py', f'{path}/checkpoints/ssd.pth',
                             datapath=path,
                             dataset=f'{dataset}_{stream}_test_{e}',
                             out=f'{path}/snapshot/result/{dataset}_{_stream}_base/{e:02d}.pkl'
                             )
                else:
                    os.makedirs(f'{path}/snapshot/result/{dataset}_{_stream}_{retrain}_{postfix}', exist_ok=True)
                    for e in range(epoch):
                        test('configs/custom/ssd_base.py', f'{path}/snapshot/models/{dataset}_{_stream}_{retrain}_{postfix}/{e}.pth',
                             datapath=path,
                             dataset=f'{dataset}_{stream}_test_{e}',
                             out=f'{path}/snapshot/result/{dataset}_{_stream}_{retrain}_{postfix}/{e:02d}.pkl'
                             )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch validation')
    parser.add_argument('--path', '-p', type=str, required=True)
    parser.add_argument('--dataset', '-d', type=str, default='detrac')
    parser.add_argument('--stream-l', '-sl', type=int, default=1)
    parser.add_argument('--stream-r', '-sr', type=int, default=12)
    parser.add_argument('--retrain-l', '-rl', type=int, default=1)
    parser.add_argument('--retrain-r', '-rr', type=int, default=6)
    parser.add_argument('--postfix', '-o', type=str, default='e40')
    parser.add_argument('--epoch', '-n', type=int, default=12)
    parser.add_argument('--val', '-v', type=ast.literal_eval, default=True)
    args = parser.parse_args()
    batch_validate(**args.__dict__)
