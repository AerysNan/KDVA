#!/usr/bin/python

import ast
import json
import argparse
from evaluate_system import evalutate_system


def range_evaluation(path, dataset, config, count, stride):
    result = []
    for i in range(count):
        result.append(evalutate_system(path, f'{dataset}_test_{i}', config, i * stride, i * stride + stride))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate system performance')
    parser.add_argument('--path', '-p', help='path to result files', type=str, required=True)
    parser.add_argument('--dataset', '-d', help='name of dataset', type=str, required=True)
    parser.add_argument('--config', '-c', help='config file path', type=str, default='/home/ubuntu/urban/configs/custom/ssd.py')
    parser.add_argument('--count', '-n', help='number of splitted ranges', type=int, default=15)
    parser.add_argument('--summary', '-s', help='whether to evaluate the whole sequence', type=ast.literal_eval, default=False)
    args = parser.parse_args()

    with open('datasets.json') as f:
        datasets = json.load(f)
    if args.dataset not in datasets:
        key = args.dataset[:args.dataset.rfind('_')]
    else:
        key = args.dataset
    print(args.dataset, key)
    n = datasets[key]['size']

    l = range_evaluation(args.path, args.dataset, args.config, args.count, n // args.count)
    for v in l:
        print(v)
    if args.summary:
        print(evalutate_system(args.path, key, args.config))
