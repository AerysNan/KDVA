from collections import defaultdict
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

val = defaultdict(lambda: [[], []])
test = defaultdict(lambda: [[], []])
key = ['020', '040', '060', '080', '100']


def plot_log(input, output):
    global val, test
    f = open(input)
    skip = False
    train_loss, test_loss, test_mAP, val_loss, val_mAP = [[], []], [[], [], [], []], [[], [], []], [[], [], [], []], [[], [], []]
    for line in f:
        if skip:
            skip = False
            continue
        try:
            o = json.loads(line)
        except:
            continue
        if 'mode' not in o:
            continue
        if o['mode'] == 'train':
            train_loss[0].append(o['epoch'])
            train_loss[1].append(o['loss'])
            skip = True
        elif o['mode'] == 'val' and 'loss' in o:
            if o['iter'] == 25:
                test_loss[0].append(o['epoch'])
                test_loss[1].append(o['loss'])
                test_loss[2].append(o['loss_cls'])
                test_loss[3].append(o['loss_bbox'])
            else:
                val_loss[0].append(o['epoch'])
                val_loss[1].append(o['loss'])
                val_loss[2].append(o['loss_cls'])
                val_loss[3].append(o['loss_bbox'])
            skip = True
        elif o['mode'] == 'val' and 'bbox_mAP' in o:
            if o['iter'] == 500:
                test_mAP[0].append(o['epoch'])
                test_mAP[1].append(o['bbox_mAP'])
                test_mAP[2].append(float(o["bbox_mAP_car"]))
            else:
                val_mAP[0].append(o['epoch'])
                val_mAP[1].append(o['bbox_mAP'])
                val_mAP[2].append(float(o["bbox_mAP_car"]))
    if len(test_mAP[2]) != 10:
        print(input)
        return
    for k in key:
        if k in output:
            for v in val_mAP[1]:
                val[k][0].append(v)
            for v in val_mAP[2]:
                val[k][1].append(v)
            for v in test_mAP[1]:
                test[k][0].append(v)
            for v in test_mAP[2]:
                test[k][1].append(v)
    plt.figure()
    plt.plot(train_loss[0], train_loss[1], color="blue", linewidth=1.0, linestyle="-")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f'{output} train loss')
    plt.savefig(f'images/train_loss/{output}.jpg')
    plt.close()

    plt.figure()
    plt.plot(test_loss[0], test_loss[1], color="blue", linewidth=1.0, linestyle="-", label='test-total')
    plt.plot(test_loss[0], test_loss[2], color="red", linewidth=1.0, linestyle="-", label='test-cls')
    plt.plot(test_loss[0], test_loss[3], color="green", linewidth=1.0, linestyle="-", label='test-bbox')
    plt.plot(val_loss[0], val_loss[1], color="orange", linewidth=1.0, linestyle="-", label='val-total')
    plt.plot(val_loss[0], val_loss[2], color="purple", linewidth=1.0, linestyle="-", label='val-cls')
    plt.plot(val_loss[0], val_loss[3], color="yellow", linewidth=1.0, linestyle="-", label='val-bbox')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f'{output} test val loss')
    plt.savefig(f'images/test_val_loss/{output}.jpg')
    plt.close()

    plt.figure()
    plt.plot(test_mAP[0], test_mAP[1], color="blue", linewidth=1.0, linestyle="-", label='test-total')
    plt.plot(test_mAP[0], test_mAP[2], color="red", linewidth=1.0, linestyle="-", label='test-classwise')
    plt.plot(val_mAP[0], val_mAP[1], color="green", linewidth=1.0, linestyle="-", label='val-total')
    plt.plot(val_mAP[0], val_mAP[2], color="orange", linewidth=1.0, linestyle="-", label='val-classwise')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("mAP")
    plt.title(f'{output} test val mAP')
    plt.savefig(f'images/test_val_mAP/{output}.jpg')
    plt.close()


parser = argparse.ArgumentParser(
    description='Analyze log files')
parser.add_argument(
    '--input', '-i', help='input log file', type=str, default=None)
parser.add_argument(
    '--output', '-o', help='output name', type=str, default=None)
args = parser.parse_args()


os.makedirs('images/train_loss', exist_ok=True)
os.makedirs('images/test_val_loss', exist_ok=True)
os.makedirs('images/test_val_mAP', exist_ok=True)


if args.input is not None and args.output is not None:
    plot_log(args.input, args.output)
else:
    m = {}
    for d in sorted([d for d in os.listdir('.') if 'tmp_' in d]):
        files = sorted([file for file in os.listdir(d) if 'json' in file])
        for i, file in enumerate(files):
            plot_log(f'{d}/{file}', f'trace {d[8]} sample {d[10: 13]} interval {i}')
