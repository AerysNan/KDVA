import os
from shutil import copyfile


def generate_sample_position(sample_count, sample_interval):
    sample_win, total = [1 for _ in range(sample_count)], sample_count
    while total < sample_interval:
        for i in range(sample_count):
            sample_win[i] += 1
            total += 1
            if total == sample_interval:
                break
    pos = [0]
    for i in range(sample_count - 1):
        pos.append(pos[-1] + sample_win[i])
    return pos


original_framerate = 25

for framerate in range(21, 25):
    for stream in range(1, 13):
        count = 0
        os.makedirs(f'data/detrac_{stream}_{framerate}fps', exist_ok=True)
        files = os.listdir(f'data/detrac_{stream}')
        poses = generate_sample_position(framerate, 25)

        for i in range(240):
            for pos in poses:
                copyfile(f'data/detrac_{stream}/{i * 25 + pos:06d}.jpg', f'data/detrac_{stream}_{framerate}fps/{count:06d}.jpg')
                count += 1
        # for pos in poses:


def print(data, path):
    with open(path, 'w') as f:
        for line in data:
            f.write('\t'.join([str(v) for v in line]))
            f.write('\n')
