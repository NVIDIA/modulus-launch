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

from typing import Optional

import torch
from torch.optim import lr_scheduler

try:  # TODO (mnabian): better handle this
    from apex import optimizers

    apex_available = True
except ImportError:
    apex_available = False


class OptimizerHandler:
    """
    Handles optimizer & scheduler initialization
    """

    def __init__(  # TODO (mnabian): add type hints
        self,
        model_parameters,
        configs,  # TODO (mnabian): get rid of configs and pass in explicit arguments
        logger=None,
    ):
        self.model_parameters = model_parameters
        self.configs = configs
        self.logger = logger
        self.apex_available = apex_available

    def get_optimizer(self):

        # Get and initialize optimizer
        if self.configs.optimizer == "FusedAdam":
            if self.apex_available:
                self.optimizer = optimizers.FusedAdam(
                    self.model_parameters,
                    betas=self.configs.betas,
                    lr=self.configs.learning_rate,
                    weight_decay=self.configs.weight_decay,
                )
                self.logger.info("using FusedAdam optimizer")
            else:
                raise ImportError(
                    "fused_adam is not available. "
                    "Install apex from https://github.com/nvidia/apex"
                )
        elif self.configs.optimizer == "FusedMixedPrecisionLamb":
            if self.apex_available:
                self.optimizer = optimizers.FusedMixedPrecisionLamb(
                    self.model_parameters,
                    betas=self.configs.betas,
                    lr=self.configs.learning_rate,
                    weight_decay=self.configs.weight_decay,
                    max_grad_norm=self.configs.optimizer_max_grad_norm,
                )
                self.logger.info("using FusedMixedPrecisionLamb optimizer")
            else:
                raise ImportError(
                    "fused_mixed_precision_lamb is not available. "
                    "Install apex from https://github.com/nvidia/apex"
                )
        elif self.configs.optimizer == "FusedLAMB":
            if self.apex_available:
                self.optimizer = optimizers.FusedLAMB(
                    self.model_parameters,
                    betas=self.configs.betas,
                    lr=self.configs.learning_rate,
                    weight_decay=self.configs.weight_decay,
                    max_grad_norm=self.configs.optimizer_max_grad_norm,
                )
                self.logger.info("using FusedLamb optimizer")
            else:
                raise ImportError(
                    "fused_lamb is not available. "
                    "Install apex from https://github.com/nvidia/apex"
                )
        elif self.configs.optimizer == "Adam":
            self.ptimizer = torch.optim.Adam(
                self.model_parameters,
                betas=self.configs.betas,
                lr=self.configs.learning_rate,
                weight_decay=self.configs.weight_decay,
            )
            self.logger.info("using Adam optimizer")
        elif self.configs.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model_parameters,
                lr=self.configs.learning_rate,
                weight_decay=self.configs.weight_decay,
            )
            self.logger.info("using SGD optimizer")
        else:
            raise NotImplementedError(
                f"Optimizer {self.configs.optimizer} is not supported."
            )
        return self.optimizer

    def get_scheduler(self):
        # Get and initialize scheduler

        if not hasattr(self, "optimizer"):
            raise ValueError("Optimizer must be initialized before scheduler.")

        if self.configs.scheduler == "ReduceLROnPlateau":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.2, patience=5, mode="min"
            )
        elif self.configs.scheduler == "CosineAnnealingLR":
            eta_min = (
                self.configs.scheduler_min_lr
                if hasattr(self.configs, "scheduler_min_lr")
                else 0.0
            )
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.configs.scheduler_T_max,
                eta_min=eta_min,
            )
        elif self.configs.scheduler == "OneCycleLR":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.configs.learnin_rate,
                total_steps=self.configs.scheduler_T_max,
                steps_per_epoch=1,
            )
        else:
            self.scheduler = None
            self.logger.warning(
                f"Scheduler {self.configs.scheduler} is not supported. No scheduler will be used."
            )

        if self.configs.lr_warmup_steps > 0:  # NOTE different from era5_wind
            warmup_scheduler = lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.configs.lr_warmup_start_factor,  # NOTE use a small value instead of 0 to avoid NaN
                end_factor=1.0,
                total_iters=self.configs.lr_warmup_steps,
            )
            self.scheduler = lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    warmup_scheduler,
                    self.scheduler,
                ],  # TODO (mnabian): add a third scheduler to support GraphCast finetuning
                milestones=[self.configs.lr_warmup_steps],
            )
        return self.scheduler
