import json
import argparse


def data_truncate(source, target, begin, end, **_):
    with open(source) as f:
        source_annotations = json.load(f)
    target_annotations = {
        'images': [],
        'annotations': [],
        'categories': source_annotations['categories'],
    }
    for image in source_annotations['images']:
        if image['id'] < begin or image['id'] >= end:
            continue
        image['id'] -= begin
        target_annotations['images'].append(image)
    for annotation in source_annotations['annotations']:
        if annotation['image_id'] < begin or annotation['image_id'] >= end:
            continue
        annotation['image_id'] -= begin
        target_annotations['annotations'].append(annotation)
    with open(target, 'w') as f:
        json.dump(target_annotations, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Move or truncate dataset')
    parser.add_argument('--source', '-s', help='source dataset', type=str, required=True)
    parser.add_argument('--target', '-t', help='target dataset', type=str, required=True)
    parser.add_argument('--begin', '-b', help='begin index of source dataset', type=int, required=True)
    parser.add_argument('--end', '-e', help='end index of source dataset (exclusive)', type=int, required=True)
    args = parser.parse_args()
    data_truncate(**args.__dict__)
