from copy import deepcopy
import json
count = 4
for stream in range(1, 3):
    for epoch in range(20):
        with open(f'data/annotations/detrac_trace_{stream}_020-500_train_{epoch}.golden.json') as f:
            annotations = json.load(f)

        replication = {
            'images': [],
            'annotations': [],
            'categories': annotations['categories'],
        }

        for image in annotations['images']:
            for i in range(count):
                _image = deepcopy(image)
                _image['id'] = _image['id'] * count + i
                replication['images'].append(_image)
        for annotation in annotations['annotations']:
            for i in range(count):
                _annotation = deepcopy(annotation)
                _annotation['id'] = _annotation['id'] * count + i
                _annotation['image_id'] = _annotation['image_id'] * count + i
                replication['annotations'].append(_annotation)
        with open(f'data/annotations/detrac_trace_{stream}_020-500-replication_train_{epoch}.golden.json', 'w') as f:
            json.dump(replication, f)
