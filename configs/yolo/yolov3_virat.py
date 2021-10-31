_base_ = './yolov3_d53_mstrain-608_273e_coco.py'

# Modify dataset related settings
dataset_type = 'CocoDataset'
data_root = 'data/'

data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/sherbrooke_train_4.json',
        img_prefix=data_root + 'sherbrooke_train_4/'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/sherbrooke_test_1.json',
        img_prefix=data_root + 'sherbrooke_test_1/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/sherbrooke_test_4.json',
        img_prefix=data_root + 'sherbrooke_test_4/')
)

load_from = 'checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
runner = dict(type='EpochBasedRunner', max_epochs=10)
