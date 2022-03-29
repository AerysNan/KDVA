import os
import json
import matplotlib.pyplot as plt


def extract_validation(stream, retrain):
    files = os.listdir(f'tmp_detrac_{stream}_{retrain:03d}_val10')
    files = [file for file in files if '.json' in file]
    l, l_downsample = [[] for _ in range(11)], [[] for _ in range(11)]
    for i, file in enumerate(files):
        f = open(f'tmp_detrac_{stream}_{retrain:03d}_val10/{file}')
        for line in f:
            o = json.loads(line)
            if 'mode' in o and o['mode'] == 'val':
                l[i].append(o['bbox_mAP_car'])
        f.close()
    files = os.listdir(f'tmp_detrac_downsample_{stream}_{retrain:03d}_val10')
    files = [file for file in files if '.json' in file]
    for i, file in enumerate(files):
        f = open(f'tmp_detrac_downsample_{stream}_{retrain:03d}_val10/{file}')
        for line in f:
            o = json.loads(line)
            if 'mode' in o and o['mode'] == 'val':
                l_downsample[i].append(o['bbox_mAP_car'])
        f.close()
    return l, l_downsample


l, l_downsample = extract_validation(1, 20)
plt.figure(figsize=(16, 9))

for i in range(11):
    plt.subplot(3, 4, i + 1)
    plt.plot(range(1, 21), l[i], linestyle='-', color='red', label='origin')
    plt.plot(range(1, 21), l_downsample[i], linestyle='-', color='blue', label='downsample')
    plt.legend()
    plt.savefig('haha.jpg')
