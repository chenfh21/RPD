"""Train semantic segmentation model."""

import argparse
from calendar import c
import os
import pdb
import time
from typing import Dict

# import git
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from callbacks import (ConfigCallback, PostprocessorCallback, VisualizerCallback, get_postprocessors, get_visualizers)
from datasets import get_data_module
from modules import get_backbone, get_criterion, model_multimetrics

import torch

def parse_args() -> Dict[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', default='log_dir/', help='Path to export dir which saves logs, metrics, etc.')
    parser.add_argument('--config', default='./config/config_deeplearn.yaml',
                        help="Path to configuration file (*.yaml)")
    parser.add_argument('--ckpt_path', default=None, help='Provide *.ckpt file to continue training.')
    parser.add_argument('--resume', default=False, action='store_true')  # implies default = False

    args = vars(parser.parse_args())

    return args


def load_config(path_to_config_file: str) -> Dict:
    assert os.path.exists(path_to_config_file)

    with open(path_to_config_file) as istream:
        config = yaml.safe_load(istream)

    return config


def main():
    args = parse_args()

    cfg = load_config(args['config'])  # 读取yaml文件(->Dict)
    # cfg['git-commit'] = get_git_commit_hash()  ## 用于上传github文件

    if cfg.get('seed') is None:
        seed_val = int(time.time())
        cfg['seed'] = seed_val
    else:
        seed_val = cfg['seed']
    seed_everything(seed_val)  # 设置随机种子，确保实验的可重复性

    datasetmodule = get_data_module(cfg)  # 封装了所有数据训练步骤
    criterion = get_criterion(cfg)

    # define backbone
    network = get_backbone(cfg)

    if (args['ckpt_path'] is not None) and (not args['resume']):
        seg_module = model_multimetrics.SegmentationNetwork(network,
                                                criterion,
                                                cfg['train']['learning_rate'],
                                                cfg['train']['weight_decay'],
                                                train_step_settings=cfg['train']['step_settings'],
                                                val_step_settings=cfg['val']['step_settings'],
                                                ckpt_path=args['ckpt_path'])
    else:
        seg_module = model_multimetrics.SegmentationNetwork(network,
                                                            criterion,
                                                            cfg['train']['learning_rate'],
                                                            cfg['train']['weight_decay'],
                                                            train_step_settings=cfg['train']['step_settings'],
                                                            val_step_settings=cfg['val']['step_settings'])

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_saver_val_loss = ModelCheckpoint(
        monitor='val_loss',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{val_loss:.4f}',
        mode='min',
        save_last=True)
    checkpoint_saver_val_mIoU = ModelCheckpoint(
        monitor='val_mIoU',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{val_mIoU:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_train_loss = ModelCheckpoint(
        monitor='train_loss',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{train_loss:.4f}',
        mode='min',
        save_last=False)
    checkpoint_saver_train_mIoU = ModelCheckpoint(
        monitor='train_mIoU',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{train_mIoU:.4f}',
        mode='max',
        save_last=False)

    checkpoint_saver_val_mPrecision = ModelCheckpoint(
        monitor='val_mPrecision',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{val_mPrecision:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_train_mPrecision = ModelCheckpoint(
        monitor='train_mPrecision',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{train_mPrecision:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_val_mF1 = ModelCheckpoint(
        monitor='val_mF1',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{val_mF1:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_train_mF1 = ModelCheckpoint(
        monitor='train_mF1',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{train_mF1:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_val_mAcc = ModelCheckpoint(
        monitor='val_mAcc',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{val_mAcc:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_train_mAcc = ModelCheckpoint(
        monitor='train_mAcc',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{train_mAcc:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_val_OverallAcc = ModelCheckpoint(
        monitor='val_OverallAcc',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{val_OverallAcc:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_train_OverallAcc = ModelCheckpoint(
        monitor='train_OverallAcc',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{train_OverallAcc:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_val_mRecall = ModelCheckpoint(
        monitor='val_mRecall',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{val_mRecall:.4f}',
        mode='max',
        save_last=False)
    checkpoint_saver_train_mRecall = ModelCheckpoint(
        monitor='train_mRecall',
        filename=cfg['experiment']['id'] + '_{epoch:02d}_{train_mRecall:.4f}',
        mode='max',
        save_last=False)


    my_checkpoint_savers = [var_value for var_name, var_value in locals().items() if
                            var_name.startswith('checkpoint_saver')]

    visualizer_callback = VisualizerCallback(get_visualizers(cfg),
                                             cfg['train']['vis_train_every_x_epochs'],
                                             cfg['val']['vis_val_every_x_epochs'])

    postprocessor_callback = PostprocessorCallback(get_postprocessors(cfg),
                                                   cfg['train']['postprocess_train_every_x_epochs'],
                                                   cfg['val']['postprocess_val_every_x_epochs'])
    config_callback = ConfigCallback(cfg)

    early_stopping = EarlyStopping(
        monitor='val_loss',  # 监控验证集损失
        patience=3,  # 连续3次检验无改善则停止
        min_delta=0.001,  # 最小改善值，小于此值视为无改善
        mode='min',  # 因为监控的是loss，所以用"min"模式（越小越好）
        verbose=True
    )

    # Setup trainer
    trainer = Trainer(
        benchmark=cfg['train']['benchmark'],
        default_root_dir=args['export_dir'],
        max_epochs=cfg['train']['max_epoch'],
        check_val_every_n_epoch=cfg['val']['check_val_every_n_epoch'],
        callbacks=[*my_checkpoint_savers,
                   lr_monitor,
                   visualizer_callback,
                   postprocessor_callback,
                   config_callback,
                   early_stopping])

    if args['ckpt_path'] is None:
        print('Train from scratch.')
        trainer.fit(seg_module, datasetmodule)
    elif (args['ckpt_path'] is not None) and (not args['resume']):
        print('Load pretrained model weights but other params (e.g. learning rate) start from scratch.')
        trainer.fit(seg_module, datasetmodule)
    elif (args['ckpt_path'] is not None) and args['resume']:
        print("Load pretrained model weights and resume training.")
        trainer.fit(seg_module, datasetmodule, ckpt_path=args['ckpt_path'])
    else:
        raise RuntimeError("Can't train any model since the settings are invalid.")


if __name__ == '__main__':
    main()


