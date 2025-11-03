import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, Callback
import torch.backends.cudnn
import yaml
from typing import Dict

from modules import get_backbone, get_criterion, module, convert_block
from datasets import get_data_module
from callbacks import *
import os


def RPD_model_deploy(model: torch.nn.Module, ckpt_dict, save_path=None, do_copy=True):
    import copy
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        save_path_convert = os.path.join(save_path, 'deploy_model.ckpt')
        ckpt_dict['state_dict'] = model.state_dict()
        torch.save(ckpt_dict, save_path_convert)
    return model


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Deploy_Convert_Multi_Testing !')

    parser.add_argument('--export_dir', metavar='SAVE', default='log_dir/deploy_convert_multi_testing',
                        help='Path to save files related to re-parameterizing multi_branch into a single branch !')
    parser.add_argument('--config', default='./config/config_deploy_convert.yaml',
                        help="Path to configuration file (*.yaml)")
    parser.add_argument('--convert_ckpt_path', type=str,
                        # default='/home/cfh/EDB/log_dir/convert_path/convert_pdc_weights.pth',
                        default='/home/cfh/EDB/log_dir/lightning_logs/version_97/checkpoints/phenobench_epoch=4095_train_mIoU=0.9074.ckpt',
                        # default='G:\BaiduNetdiskDownload\EDB\log_dir\convert_path\convert_pdc_weights.ckpt',
                        help='Provide converted *.ckpt file to load.')

    args = vars(parser.parse_args())

    return args


def config(path_to_config_file) -> Dict:
    assert os.path.exists(path_to_config_file)

    with open(path_to_config_file, 'r') as istream:
        config = yaml.safe_load(istream)

    return config


def main():
    args = parse_args()

    cfg = config(args['config'])

    # define train_model
    if cfg['backbone']['convert']:
        train_model = get_backbone(cfg)
    if os.path.isfile(args['convert_ckpt_path']):
        print("=> loading checkpoint '{}'".format(args['convert_ckpt_path']))
        ckpt_dict = torch.load(
            args['convert_ckpt_path'])  # , map_location='gpu' if torch.cuda.is_available() else 'cpu')
        if 'state_dict' in ckpt_dict:
            state_dict = ckpt_dict['state_dict']
        elif 'model' in ckpt_dict:
            state_dict = ckpt_dict['model']
        elif 'network' in ckpt_dict:
            state_dict = ckpt_dict['network']

        # check if there are any spaces in the state_dict keys.
        ckpt = {k.replace('module.', ''): v for k, v in state_dict.items()}  # strip the names

        train_model.load_state_dict(ckpt, strict=False)
        ckpt_dict['state_dict'] = ckpt_dict
    else:
        print("=> no checkpoint found at '{}'".format(args['convert_ckpt_path']))
    train_model.eval()
    RPD_model_deploy(train_model, ckpt_dict, save_path=args['export_dir'])


if __name__ == '__main__':

    main()

