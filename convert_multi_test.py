import argparse

import pytorch_lightning
from pytorch_lightning import LightningModule, Trainer, Callback
import torch.backends.cudnn
import yaml
from typing import Dict

from modules import get_backbone, get_criterion, module, convert_block
from datasets import get_data_module
from callbacks import *
import os


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Convert_Multi_Testing !')

    parser.add_argument('--export_dir', default='log_dir/convert_multi_testing',
                        help='Path to save files related to converting multi_branch into a single branch !')
    parser.add_argument('--config', default='./config/config_convert.yaml',
                        help="Path to configuration file (*.yaml)")
    parser.add_argument('--ckpt_path', type=str,
                        default='./log_dir/.../checkpoints/XXX.ckpt',
                        help='Provide *.ckpt file to convert.')
    parser.add_argument('--deploy_model',
                        default='./log_dir/deploy_convert_multi_testing/deploy_model.ckpt',
                        help='Path to deploy file (*.ckpt/pth)')

    args = vars(parser.parse_args())

    return args


def load_config(path_to_config_file: str) -> Dict:
    assert os.path.exists(path_to_config_file)

    with open(path_to_config_file) as istream:
        config = yaml.safe_load(istream)

    return config


def main():
    args = parse_args()

    cfg = load_config(args['config'])

    datasetmodule = get_data_module(cfg)
    criterion = get_criterion(cfg)

    print('deploy', cfg['backbone']['deploy'], 'convert', cfg['backbone']['convert'])

    # define backbone
    network = get_backbone(cfg)

    if cfg['backbone']['deploy']:
        print("Deploy model !")
        convert_path = torch.load(args['deploy_model'])
    else:
        if cfg['backbone']['convert']:
            print('This is just the process of re-parameterized weights!')
            ckpt_dict = torch.load(args['ckpt_path'])
            network.load_state_dict(convert_block(ckpt_dict['state_dict'], 'pdc'), strict=False)
            ckpt_dict['state_dict'] = convert_block(ckpt_dict['state_dict'], 'pdc')

            convert_path = '/home/cfh/EDB/log_dir/convert_path/convert_pdc_weights.pth'

            torch.save(ckpt_dict, convert_path)
            print('save the converted weights ...')
        else:
            print('Using original model and weights !')
            convert_path = torch.load(args['ckpt_path'])

    seg_module = module.SegmentationNetwork(network,
                                            criterion,
                                            cfg['train']['learning_rate'], 
                                            cfg['train']['weight_decay'],
                                            train_step_settings=cfg['train']['step_settings'],
                                            val_step_settings=cfg['val']['step_settings'],
                                            )

    # # Add callbacks
    visualizer_callback = VisualizerCallback(get_visualizers(cfg))
    postprocessor_callback = PostprocessorCallback(get_postprocessors(cfg))
    config_callback = ConfigCallback(cfg)

    # Setup trainer
    trainer = Trainer(default_root_dir=args['export_dir'],
                      max_epochs=cfg['train']['max_epoch'],
                      devices=cfg['val']['devices'],
                      num_nodes=cfg['val']['num_nodes'],
                      callbacks=[visualizer_callback,
                                 postprocessor_callback,
                                 config_callback])

    trainer.validate(seg_module,
                     datasetmodule,
                     ckpt_path=convert_path)


if __name__ == '__main__':

    main()

