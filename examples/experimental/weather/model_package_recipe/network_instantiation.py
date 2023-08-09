# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import hydra
import wandb
import matplotlib.pyplot as plt
from functools import partial

from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler
from omegaconf import DictConfig

from modulus.registry import ModelRegistry
from modulus.models import Module

from modulus.launch.logging import (
    LaunchLogger,
    PythonLogger,
    RankZeroLoggingWrapper,
    initialize_mlflow,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # instantiate the model (Replace if statement with this)
    registry = ModelRegistry()
    assert (
        cfg.model_name in registry.list_models()
    ), f"Model {cfg.model_name} not found in Modulus registry"
    model = Module.instantiate({"__name__": cfg.model_name, "__args__": cfg.model})
    print(model)


if __name__ == "__main__":
    main()
