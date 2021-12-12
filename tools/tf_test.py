import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse
import pickle
import time
import json


parser = argparse.ArgumentParser(
    description='Tensorflow Test')
parser.add_argument('--dataset', '-d', help='input dataset', type=str, required=True)
parser.add_argument('--output', '-o', help='output file path', type=str, required=True)
parser.add_argument('--model', '-m', help='EfficientDet version', type=int, default=0)
args = parser.parse_args()

detector = hub.load(f'https://tfhub.dev/tensorflow/efficientdet/d{args.model}/1')

with open(f'classes_tf.json') as f:
    classes = json.load(f)
with open(f'datasets.json') as f:
    datasets = json.load(f)
mapping = {}
for i, cat in enumerate(classes):
    mapping[cat['id']] = i

n = datasets[args.dataset]['size']

results = []

scale_x, scale_y = 1920, 1080

start = time.time()
previous, previous_time = 0, time.time()
for i in range(4224, 4225):
    img_raw = tf.io.read_file(f'data/{args.dataset}/{i:06d}.jpg')
    img_tensor = tf.expand_dims(tf.image.decode_image(img_raw), axis=0)
    detector_output = detector(img_tensor)
    result = [[] for _ in range(80)]
    for j, box in enumerate(detector_output['detection_boxes'][0]):
        class_id = int(detector_output["detection_classes"][0, j].numpy())
        box_tolist = box.numpy().tolist()
        box_tolist[0] *= scale_y
        box_tolist[1] *= scale_x
        box_tolist[2] *= scale_y
        box_tolist[3] *= scale_x
        box_tolist[0], box_tolist[1] = box_tolist[1], box_tolist[0]
        box_tolist[2], box_tolist[3] = box_tolist[3], box_tolist[2]
        box_tolist.append(detector_output["detection_scores"][0, j].numpy())
        if class_id not in mapping:
            print(f'!!! {args.dataset} {i} {class_id}')
            continue
        result[mapping[class_id]].append(box_tolist)
    for j in range(80):
        result[j] = np.array(result[j], dtype=np.float32)
        if result[j].shape == (0, ):
            result[j] = np.zeros((0, 5))
    results.append(result)
    if i > 0 and i % 100 == 0:
        print(f'Current frame: {i}; elapsed time: {time.time() - start:.3f}s; throughput: {(i + 1 - previous) / (time.time() - previous_time):.2f} FPS')
        previous = i + 1
        previous_time = time.time()

# with open(args.output, 'wb') as f:
#     pickle.dump(results, f)
