#!/usr/bin/python

import ast
import json
import argparse
from evaluate_from_file import evaluate_from_file


def range_evaluation(result, dataset, count, postfix, config, gt):
    l = []
    for i in range(count):
        path = f'/{postfix}' if postfix is not None else ''
        l.append(evaluate_from_file(f'snapshot/result/{result}{path}/{i:02d}.pkl', f'data/annotations/{dataset}_test_{i}.{"gt" if gt else "golden"}.json', config))
    return l


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate system performance')
    parser.add_argument('--result', '-r', help='result name', type=str, required=True)
    parser.add_argument('--dataset', '-d', help='name of dataset', type=str, required=True)
    parser.add_argument('--config', '-c', help='config file path', type=str, default='configs/custom/ssd_base.py')
    parser.add_argument('--count', '-n', help='number of splitted ranges', type=int, default=15)
    parser.add_argument('--summary', '-s', help='whether to evaluate the whole sequence', type=ast.literal_eval, default=False)
    parser.add_argument('--postfix', '-p', help='postfix', type=str, default=None)
    parser.add_argument('--gt', '-g', help='use ground truth for evaluation', type=ast.literal_eval, default=True)
    args = parser.parse_args()

    with open('datasets.json') as f:
        datasets = json.load(f)
    if args.dataset not in datasets:
        key = args.dataset[:args.dataset.rfind('_')]
    else:
        key = args.dataset
    n = datasets[key]['size']
    l = range_evaluation(args.result, args.dataset, args.count, args.postfix, args.config, args.gt)
    if args.summary:
        path = f'_{args.postfix}' if args.postfix is not None else ''
        l.append(evaluate_from_file(f'snapshot/merge/{args.result}{path}.pkl', f'data/annotations/{key}.{"gt" if args.gt else "golden"}.json', args.config))
    for v in l:
        print(v['bbox_mAP'])
    print('classwise')
    for v in l:
        print(v["bbox_mAP_car"])
