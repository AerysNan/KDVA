import copy
import pickle
import argparse
import numpy as np

from mmcv import Config
from mmdet.datasets import build_dataset


def main():
    parser = argparse.ArgumentParser(
        description='Fake replay')
    parser.add_argument('--config', '-c', type=str, default='/home/ubuntu/urban/configs/custom/ssd.py', help='test config file path')
    args = parser.parse_args()

    datasets = []
    m = {
        0: "sub_1",
        1: "sub_3",
        2: "sub_6",
        3: "sub_8"
    }

    for k in range(4):
        cfg = Config.fromfile(args.config)
        cfg.data.test.ann_file = f'data/annotations/{m[k]}_fake.json'
        cfg.data.test.img_prefix = f'data/{m[k]}'
        dataset = build_dataset(cfg.data.test)
        datasets.append(dataset)

    results = [[], [], [], []]
    previouses = []
    for k in range(4):
        with open(f'snapshot/result/{m[k]}/0.pkl', 'rb') as f:
            previous = pickle.load(f)
        previouses.append(previous)

    with open('data/gradient.pkl', 'rb') as f:
        gradients = pickle.load(f)
    fpses = [3, 3, 3, 3]
    print("start replay")

    for i in range(36):
        gradient = []
        for k in range(4):
            gradient.append(gradients[fpses[k] - 1][i][k])
        thief, victim = np.argmax(gradient), np.argmin(gradient)
        if not fpses[victim] == 1 and not fpses[thief] == 5:
            fpses[victim] -= 1
            fpses[thief] += 1
        for k in range(4):
            begin, end, stride = i * 250, i * 250 + 250, 60 // fpses[k]
            for j in range(begin, end):
                if j % stride == 0:
                    with open(f'snapshot/result/{m[k]}/{j}.pkl', 'rb') as f:
                        o = pickle.load(f)
                        results[k].append(o)
                        previouses[k] = o
                else:
                    results[k].append(copy.deepcopy(previouses[k]))
    print("start evaluation")
    for k in range(4):
        print(datasets[k].evaluate(results[k], metric='bbox'))


if __name__ == '__main__':
    main()
