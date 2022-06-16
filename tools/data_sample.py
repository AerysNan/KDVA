import json
import argparse


def downsample_dataset(input_file, output_file, rate, **_):
    with open(input_file) as f:
        annotations = json.load(f)
    annotations_downsampled = {'images': [], 'annotations': [], 'categories': annotations['categories']}
    for image in annotations['images']:
        if image['id'] % rate == 0:
            image['id'] //= rate
            annotations_downsampled['images'].append(image)
    for annotation in annotations['annotations']:
        if annotation['image_id'] % rate == 0:
            annotation['image_id'] //= rate
            annotations_downsampled['annotations'].append(annotation)
    with open(output_file, 'w') as f:
        json.dump(annotations_downsampled, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downsample dataset')
    parser.add_argument('--input-file', '-i', type=str, required=True, help='Input dataset file')
    parser.add_argument('--output-file', '-o', type=str, required=True, help='Output dataset file')
    parser.add_argument('--rate', '-r', type=int, default=10, help='sampling rate')
    args = parser.parse_args()
    downsample_dataset(**args.__dict__)
