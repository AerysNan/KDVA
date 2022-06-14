import json
import argparse


def downsample_dataset(input, output, rate, **_):
    with open(f'data/annotations/{input}.gt.json') as f:
        annotations_gt = json.load(f)
    with open(f'data/annotations/{input}.golden.json') as f:
        annotations_golden = json.load(f)
    annotations_gt_downsampled = {'images': [], 'annotations': [], 'categories': annotations_gt['categories'], 'ignored_regions': []}
    annotations_golden_downsampled = {'images': [], 'annotations': [], 'categories': annotations_golden['categories']}
    for image in annotations_gt['images']:
        if image['id'] % rate == 0:
            image['id'] //= rate
            annotations_gt_downsampled['images'].append(image)
    for image in annotations_golden['images']:
        if image['id'] % rate == 0:
            image['id'] //= rate
            annotations_golden_downsampled['images'].append(image)
    for annotation in annotations_gt['annotations']:
        if annotation['image_id'] % rate == 0:
            annotation['image_id'] //= rate
            annotations_gt_downsampled['annotations'].append(annotation)
    for annotation in annotations_golden['annotations']:
        if annotation['image_id'] % rate == 0:
            annotation['image_id'] //= rate
            annotations_golden_downsampled['annotations'].append(annotation)
    if 'region' in annotations_gt:
        for region in annotations_gt['ignored_regions']:
            region['begin'] //= rate
            region['end'] //= rate
            annotations_gt_downsampled['ignored_regions'].append(region)
    with open(f'data/annotations/{output}.gt.json', 'w') as f:
        json.dump(annotations_gt_downsampled, f)
    with open(f'data/annotations/{output}.golden.json', 'w') as f:
        json.dump(annotations_golden_downsampled, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downsample dataset')
    parser.add_argument('--input', '-i', type=str, required=True, help='input dataset')
    parser.add_argument('--output', '-o', type=str, required=True, help='output dataset')
    parser.add_argument('--rate', '-r', type=int, default=10, help='sampling rate')
    args = parser.parse_args()
    downsample_dataset(**args.__dict__)
