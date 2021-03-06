# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger, get_device


def train(config, work_dir, train_anno_file=None, train_img_prefix=None, val_anno_file=None, val_img_prefix=None, load_from=None, max_epochs=None, resume_from=None, no_validate=True, gpus=None, gpu_ids=None, seed=None, deterministic=False, cfg_options=None, launcher='none',  **_):
    cfg = Config.fromfile(config) if type(config) == str else config
    cfg.device = get_device()
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if work_dir is not None:
        # update configs according to CLI args if work_dir is not None
        cfg.work_dir = work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config))[0])
    if resume_from is not None:
        cfg.resume_from = resume_from
    if gpu_ids is not None:
        cfg.gpu_ids = gpu_ids
    else:
        cfg.gpu_ids = range(1) if gpus is None else range(gpus)

    # init distributed env first, since logger depends on the dist info.
    if launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
    # dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    # logger.info(f'Distributed training: {distributed}')
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if seed is not None:
        # logger.info(f'Set random seed to {seed}, '
        # f'deterministic: {deterministic}')
        set_random_seed(seed, deterministic=deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    # overload configuration file
    if train_anno_file is not None:
        cfg.data.train.ann_file = train_anno_file
    if train_img_prefix is not None:
        cfg.data.train.img_prefix = train_img_prefix
    if val_anno_file is not None:
        cfg.data.val.ann_file = val_anno_file
    if val_img_prefix is not None:
        cfg.data.val.img_prefix = val_img_prefix
    no_validate = no_validate or cfg.data.val.ann_file is None and cfg.data.val.img_prefix is None
    if load_from is not None:
        cfg.load_from = load_from
    if max_epochs is not None:
        cfg.runner.max_epochs = max_epochs
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) >= 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if len(cfg.workflow) >= 3:
        test_dataset = copy.deepcopy(cfg.data.test)
        test_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(test_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_true', help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--options', nargs='+', action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                        'in xxx=yyy format will be merged into config file (deprecate), '
                        'change to --cfg-options instead.')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                        'in xxx=yyy format will be merged into config file. If the value to '
                        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                        'Note that the quotation marks are necessary and that no white space '
                        'is allowed.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int, help='number of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+', help='ids of gpus to use (only applicable to non-distributed training)')
    # custom arguments
    parser.add_argument('--train-anno-file', '-ta', help='customized train annotation file', type=str, default=None)
    parser.add_argument('--train-img-prefix', '-ti', help='customized train dataset path', type=str, default=None)
    parser.add_argument('--val-anno-file', '-va', help='customized val annotation file', type=str, default=None)
    parser.add_argument('--val-img-prefix', '-vi', help='customized val dataset path', type=str, default=None)
    parser.add_argument('--load-from', help='load from checkpoint file path', type=str, default=None)
    parser.add_argument('--max-epochs', help='maximum training epochs', type=int, default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options
    train(**args.__dict__)
