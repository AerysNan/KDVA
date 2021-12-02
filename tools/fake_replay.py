import copy
import pickle
import argparse

from mmcv import Config
from mmdet.datasets import build_dataset


def main():
    parser = argparse.ArgumentParser(
        description='Fake replay')
    parser.add_argument('--config', '-c', type=str, default='/home/ubuntu/urban/configs/custom/ssd.py', help='test config file path')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='name of dataset')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.data.test.ann_file = f'data/annotations/{args.dataset}_fake.json'
    cfg.data.test.img_prefix = f'data/{args.dataset}'
    dataset = build_dataset(cfg.data.test)

    m = {
        "sub_1": 0,
        "sub_3": 1,
        "sub_6": 2,
        "sub_8": 3
    }
    ID = m[args.dataset]

    thief = [2, 2, 1, 2, 3, 2, 2, 1, 2, 3, 2, 2, 3, 3, 2, 1, 1, 1, 3, 3, 3, 0,
             1, 3, 2, 2, 0, 3, 3, 1, 2, 2, 1, 1, 0]

    victim = [1, 3, 2, 3, 0, 3, 0, 2, 1, 1, 3, 1, 2, 2, 3, 3, 3, 2, 2, 0, 0, 3,
              0, 1, 3, 0, 2, 1, 1, 0, 1, 0, 2, 0, 1]

    result = []
    with open(f'snapshot/result/{args.dataset}/0.pkl', 'rb') as f:
        previous = pickle.load(f)
    for i in range(36):
        fps = 3
        if i > 0:
            if ID == thief[i - 1]:
                fps += 1
            elif ID == victim[i - 1]:
                fps -= 1
        begin, end, stride = i * 250, i * 250 + 250, 60 // fps
        for j in range(begin, end):
            if j % stride == 0:
                with open(f'snapshot/result/{args.dataset}/{j}.pkl', 'rb') as f:
                    o = pickle.load(f)
                    result.append(o)
                    previous = o
            else:
                result.append(copy.deepcopy(previous))
    return dataset.evaluate(result, metric='bbox')


if __name__ == '__main__':
    print(main())
