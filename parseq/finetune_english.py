#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from pathlib import PurePath
import yaml

from omegaconf import DictConfig, open_dict
import hydra
from hydra.core.hydra_config import HydraConfig
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem
from strhub.models.utils import get_pretrained_weights
from strhub.models.utils import load_from_checkpoint
from strhub.models.parseq.system import PARSeq

def _get_config_english(experiment: str, **kwargs):
    """Emulates hydra config resolution"""
    root = PurePath(__file__).parent
    with open(root / 'configs/english.yaml', 'r') as f:
        config = yaml.load(f, yaml.Loader)['model']
    with open(root / f'configs/charset/94_full.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader)['model'])
    with open(root / f'configs/experiment/{experiment}.yaml', 'r') as f:
        exp = yaml.load(f, yaml.Loader)
    # Apply base model config
    model = exp['defaults'][0]['override /model']
    with open(root / f'configs/model/{model}.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader))
    # Apply experiment config
    if 'model' in exp:
        config.update(exp['model'])
    config.update(kwargs)
    # Workaround for now: manually cast the lr to the correct type.
    config['lr'] = float(config['lr'])
    return config

@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    with open_dict(config):
        # Resolve absolute path to data.root_dir
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)
        # Special handling for GPU-affected config
        # gpus = config.trainer.get('gpus', 0)
        #if gpus:
            # Use mixed-precision training
            #config.trainer.precision = 16

    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    model: BaseSystem = hydra.utils.instantiate(config.model)
    # If specified, use pretrained weights to initialize the model
    # if config.pretrained is not None:
    #     model.load_state_dict(get_pretrained_weights(config.pretrained))
    print(summarize(model, max_depth=1 if model.hparams.name.startswith('parseq') else 2))
    print(f"NOTIFICATION:: checkpoint path {config.checkpoint_finetune}")


    # model = load_from_checkpoint(config.checkpoint_finetune, **(config.model))
    
    model = PARSeq(**(_get_config_english("parseq", **config)))
    model.load_state_dict(torch.load(config.checkpoint_finetune))
    if config.trainer.gpus == 0:
        model = model.to("cpu")
    else:
        model = model.to("cuda")
    
    
    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    checkpoint = ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=3, save_last=True,
                                 filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}')
    # swa = StochasticWeightAveraging(swa_epoch_start=0.75)
    cwd = HydraConfig.get().runtime.output_dir if config.ckpt_path is None else \
        str(Path(config.ckpt_path).parents[1].absolute())
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=TensorBoardLogger(cwd, '', '.'),
                                               strategy=None, enable_model_summary=False,
                                               callbacks=[checkpoint])
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)


if __name__ == '__main__':
    main()
