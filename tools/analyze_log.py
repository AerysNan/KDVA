import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

m = np.zeros((8 * 19, 10), dtype=np.double)
t = 0


def plot_log(input, output):
    global m, t
    f = open(input)
    train_loss, test_loss, test_mAP = [[], []], [[], [], [], []], [[], [], []]
    for line in f:
        try:
            o = json.loads(line)
        except:
            continue
        if 'mode' not in o:
            continue
        if o['mode'] == 'train':
            train_loss[0].append(o['epoch'])
            train_loss[1].append(o['loss'])
        elif o['mode'] == 'val' and 'loss' in o:
            test_loss[0].append(o['epoch'])
            test_loss[1].append(o['loss'])
            test_loss[2].append(o['loss_cls'])
            test_loss[3].append(o['loss_bbox'])
        elif o['mode'] == 'val' and 'bbox_mAP' in o:
            test_mAP[0].append(o['epoch'])
            test_mAP[1].append(o['bbox_mAP'])
            test_mAP[2].append(float(o['classwise'][2][1]))
    if len(test_mAP[2]) != 10:
        print(input)
        return
    m[t, :] = test_mAP[2]
    t += 1
    # plt.figure()
    # plt.plot(train_loss[0], train_loss[1], color="blue", linewidth=1.0, linestyle="-")
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.savefig(f'images/train_loss/{output}.jpg')
    # plt.close()
    # plt.figure()
    # plt.plot(test_loss[0], test_loss[1], color="blue", linewidth=1.0, linestyle="-", label='total')
    # plt.plot(test_loss[0], test_loss[2], color="red", linewidth=1.0, linestyle="-", label='cls')
    # plt.plot(test_loss[0], test_loss[3], color="green", linewidth=1.0, linestyle="-", label='bbox')
    # plt.legend()
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.savefig(f'images/test_loss/{output}.jpg')
    # plt.close()
    # plt.figure()
    # plt.plot(test_mAP[0], test_mAP[1], color="blue", linewidth=1.0, linestyle="-", label='total')
    # plt.plot(test_mAP[0], test_mAP[2], color="red", linewidth=1.0, linestyle="-", label='classwise')
    # plt.legend()
    # plt.xlabel("epoch")
    # plt.ylabel("mAP")
    # plt.savefig(f'images/test_mAP/{output}.jpg')
    # plt.close()


parser = argparse.ArgumentParser(
    description='Analyze log files')
parser.add_argument(
    '--input', '-i', help='input log file', type=str, default=None)
parser.add_argument(
    '--output', '-o', help='output name', type=str, default=None)
args = parser.parse_args()


os.makedirs('images/train_loss', exist_ok=True)
os.makedirs('images/test_loss', exist_ok=True)
os.makedirs('images/test_mAP', exist_ok=True)

if args.input is not None and args.output is not None:
    plot_log(args.input, args.output)
else:
    for d in sorted([d for d in os.listdir('.') if 'tmp_' in d]):
        files = sorted([file for file in os.listdir(d) if 'json' in file])
        for i, file in enumerate(files):
            plot_log(f'{d}/{file}', f'{d[4:]}_epoch_{i}')
