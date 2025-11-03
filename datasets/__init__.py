from typing import Dict

import pytorch_lightning as pl

from pdc_datasets.pdc import PDCModule
from pdc_datasets.GrowliFlower import GrowliFlowerModule

def get_data_module(cfg: Dict) -> pl.LightningDataModule:
    dataset_name = cfg['data']['name']
    if dataset_name == 'phenobench':  # phenobench'CWFID'
        return PDCModule(cfg)

    elif dataset_name == 'GrowliFlower':
        return GrowliFlowerModule(cfg)

    elif dataset_name == 'CoFly-WeedDB':
        return PDCModule(cfg)

    else:
        assert False, f'There is no parser for: {dataset_name}.'
