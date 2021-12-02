#!/usr/bin/python

import argparse
from evaluate_system import evalutate_system


def range_evaluation(id, dataset, config, count, postfix):
    result = []
    for i in range(count):
        result.append(evalutate_system(id, f'{dataset}_test_{i}_{postfix}', config, False, i * 250, i * 250 + 250))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate system performance')
    parser.add_argument('--id', '-i', help='ID of video source to be evaluated', type=str, required=True)
    parser.add_argument('--dataset', '-d', help='name of dataset', type=str, required=True)
    parser.add_argument('--config', '-c', help='config file path', type=str, default='/home/ubuntu/urban/configs/custom/ssd.py')
    parser.add_argument('--count', '-n', help='number of splitted ranges', type=int, default=36)
    parser.add_argument('--postfix', '-p', help='postfix', type=str, default='250_10')
    args = parser.parse_args()
    l = range_evaluation(args.id, args.dataset, args.config, args.count, args.postfix)
    for v in l:
        print(v)
    print(f'mAP = {evalutate_system(args.id, args.dataset, args.config, True)}')
