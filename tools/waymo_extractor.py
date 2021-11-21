import os
import argparse
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

parser = argparse.ArgumentParser(
    description='Decompress Waymo dataset')
parser.add_argument(
    '--path', '-p', help='path to dataset file', type=str, required=True)
parser.add_argument(
    '--out', '-o', help='output dataset name', type=str, required=True)
args = parser.parse_args()

os.mkdir(f'data/{args.out}')

dataset = tf.data.TFRecordDataset(args.path)
for i, data in enumerate(dataset):
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    for j, image in enumerate(frame.images):
        with open(f'data/{args.out}/{i:06d}.jpg', 'wb') as f:
            f.write(image.image)
            break
