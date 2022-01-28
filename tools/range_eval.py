#!/usr/bin/python

import ast
import json
import argparse
from evaluate_from_file import evaluate_from_file


def range_evaluation(result, dataset, count):
    l = []
    for i in range(count):
        l.append(evaluate_from_file(f'snapshot/result/{result}/{i:02d}.pkl', f'data/annotations/{dataset}_test_{i}.gt.json'))
    return l


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate system performance')
    parser.add_argument('--result', '-r', help='result name', type=str, required=True)
    parser.add_argument('--dataset', '-d', help='name of dataset', type=str, required=True)
    parser.add_argument('--config', '-c', help='config file path', type=str, default='configs/custom/ssd.py')
    parser.add_argument('--count', '-n', help='number of splitted ranges', type=int, default=15)
    parser.add_argument('--summary', '-s', help='whether to evaluate the whole sequence', type=ast.literal_eval, default=False)
    args = parser.parse_args()

    with open('datasets.json') as f:
        datasets = json.load(f)
    if args.dataset not in datasets:
        key = args.dataset[:args.dataset.rfind('_')]
    else:
        key = args.dataset
    n = datasets[key]['size']
    l = range_evaluation(args.result, args.dataset, args.count)
    if args.summary:
        l.append(evaluate_from_file(f'snapshot/merge/{args.result}.pkl', f'data/annotations/{key}.gt.json'))
    for v in l:
        print(v['bbox_mAP'])
