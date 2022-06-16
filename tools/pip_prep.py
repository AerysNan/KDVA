from shutil import copyfile
from os.path import join as opj

from util.model_test import test
from util.anno_from_imgs import anno_from_imgs
from util.anno_from_result import anno_from_result

import argparse
import config
import os


def pip_preprocess(dataset, anno_threshold=0.5, base_file=None, result_file=None, anno_file=None, **kwargs):
    if 'anno_template' in kwargs and type(kwargs['anno_template']) == str:
        anno_file = kwargs['anno_template'].format(dataset)
    if 'base_template' in kwargs and type(kwargs['base_template']) == str:
        base_file = kwargs['base_template'].format(dataset)
    if 'result_template' in kwargs and type(kwargs['result_template']) == str:
        result_file = kwargs['result_template'].format(dataset)

    o_anno_path = opj(os.environ['AMLT_OUTPUT_DIR'], config.O_ANNO_DIR)
    o_log_path = opj(os.environ['AMLT_OUTPUT_DIR'], config.O_LOG_DIR)

    os.makedirs(o_anno_path, exist_ok=True)
    os.makedirs(o_log_path, exist_ok=True)

    i_data_path = opj(os.environ['AMLT_DATA_DIR'], config.I_DATA_DIR)
    i_anno_path = opj(os.environ['AMLT_DATA_DIR'], config.I_ANNO_DIR)
    i_model_path = opj(os.environ['AMLT_DATA_DIR'], config.I_MODEL_DIR)
    i_snap_path = opj(os.environ['AMLT_DATA_DIR'], config.I_SNAP_DIR)

    if anno_file is not None:
        print('Annotation file provided, skip generation!')
        copyfile(opj(i_anno_path, anno_file), opj(o_anno_path, f'{dataset}.json'))
    else:
        if base_file is not None:
            print('Dataset description file provided, skip generation!')
            copyfile(opj(i_anno_path, base_file), opj(o_anno_path, f'{dataset}.b.json'))
        else:
            print('Start generating dataset description file...')
            anno_from_imgs(
                img_path=opj(i_data_path, dataset),
                classes='classes.dat',
                out=opj(o_anno_path, f'{dataset}.b.json'),
                prefix=dataset
            )
            print('Dataset description file generated!')
        if result_file is not None:
            print('Expert model inference result provided, skip inference!')
            copyfile(opj(i_snap_path, result_file), opj(o_log_path, f'{dataset}.r.pkl'))
        else:
            print('Start inferencing with expert model...')
            test(
                config='configs/custom/rcnn_amlt.py',
                checkpoint=opj(i_model_path, config.TEACHER_MODEL),
                out=opj(o_log_path, f'{dataset}.r.pkl'),
                anno_file=opj(o_anno_path, f'{dataset}.b.json'),
                img_prefix=i_data_path
            )
            print('Inference finished!')
        print('Start generating annotation file...')
        anno_from_result(
            base_file=opj(i_anno_path, f'{dataset}.b.json'),
            result_file=opj(o_log_path, f'{dataset}.r.pkl'),
            output_file=opj(o_anno_path, f'{dataset}.json'),
            threshold=anno_threshold
        )
        print('Annotation file generated!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing pipeline')
    parser.add_argument('--dataset', '-d', help='dataset name', type=str, required=True)
    parser.add_argument('--anno-threshold', '-t', help='annotation threshold', type=float, default=0.5)
    parser.add_argument('--anno-file', '-af', help='annotation file', type=str, default=None)
    parser.add_argument('--base-file', '-bf', help='base file', type=str, default=None)
    parser.add_argument('--result-file', '-rf', help='result file', type=str, default=None)
    args = parser.parse_args()
    pip_preprocess(**args.__dict__)
