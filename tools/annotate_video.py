from mmdet.apis import init_detector
import argparse
import pickle
import shutil
import numpy as np
import cv2
import os

parser = argparse.ArgumentParser(description='Object detection')
parser.add_argument('--dataset', '-d', type=str, required=True,
                    help='dataset for detection')
parser.add_argument('--config', '-c', type=str, default='configs/custom/ssd.py',
                    help='configuration path of the model')
parser.add_argument('--model', '-m', type=str, default='checkpoints/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth',
                    help='checkpoint path of the model')
parser.add_argument('--result', '-r', type=str, required=True,
                    help='result file path')
parser.add_argument('--begin', '-b', type=int, required=True,
                    help='begin index')
parser.add_argument('--end', '-e', type=int, required=True,
                    help='end index')
parser.add_argument('--output', '-o', type=str, required=True, help="output name")
args = parser.parse_args()

model = init_detector(args.config, args.model)

files = os.listdir(f'data/{args.dataset}')
files.sort()

with open(args.result, 'rb') as f:
    result = pickle.load(f)

print('Annotate video ...')
os.makedirs(args.output)
for i, file in enumerate(files):
    if i < args.begin:
        continue
    if i >= args.end:
        break
    print(i)
    image = f'data/{args.dataset}/{file}'
    model.show_result(image, result[i], out_file=f'{args.output}/{i:06d}.png')

# print('Creating result ...')
# videoWriter = cv2.VideoWriter(f'{args.output}.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30, (1920 * 2, 1080))
# for i in range(args.begin, args.end):
#     print(i)
#     img1 = cv2.imread(f'tmp_{args.output}_1/{i:06d}.png')
#     img2 = cv2.imread(f'tmp_{args.output}_2/{i:06d}.png')
#     img = np.hstack((img1, img2))
#     videoWriter.write(img)

# videoWriter.release()

# shutil.rmtree(tmpdirname_1)
# shutil.rmtree(tmpdirname_2)
