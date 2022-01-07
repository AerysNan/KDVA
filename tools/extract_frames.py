import os
import cv2
import argparse

parser = argparse.ArgumentParser(
    description='Extract frames from video')
parser.add_argument(
    '--path', '-p', help='path to video file', type=str, required=True)
parser.add_argument(
    '--out', '-o', help='output directory', type=str, required=True)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

video = cv2.VideoCapture(args.path)
success, image = video.read()
count = 0
while success:
    cv2.imwrite(f'{args.out}/{count:06d}.jpg', image)
    print('Write: ', count)
    success, image = video.read()
    count += 1
