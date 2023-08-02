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

from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler
from omegaconf import DictConfig

from modulus.experimental.models.afno import AFNO
from modulus.experimental.models.sfno.sfnonet import (
    SphericalFourierNeuralOperatorNet as SFNO,
)
from modulus.datapipes.climate import ERA5HDF5Datapipe
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad

from modulus.launch.logging import (
    LaunchLogger,
    PythonLogger,
    RankZeroLoggingWrapper,
    initialize_mlflow,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint

try:
    from apex import optimizers

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


def loss_func(x, y, p=2.0):
    yv = y.reshape(x.size()[0], -1)
    xv = x.reshape(x.size()[0], -1)
    diff_norms = torch.linalg.norm(xv - yv, ord=p, dim=1)
    y_norms = torch.linalg.norm(yv, ord=p, dim=1)

    return torch.mean(diff_norms / y_norms)


@torch.no_grad()
def validation_step(eval_step, model, datapipe, channels=[0, 1], epoch=0):
    loss_epoch = 0
    num_examples = 0  # Number of validation examples
    # Dealing with DDP wrapper
    if hasattr(model, "module"):
        model = model.module
    model.eval()
    for i, data in enumerate(datapipe):
        invar = data[0]["invar"].detach()
        outvar = data[0]["outvar"].cpu().detach()
        predvar = torch.zeros_like(outvar)

        for t in range(outvar.shape[1]):
            output = eval_step(model, invar)
            invar.copy_(output)
            predvar[:, t] = output.detach().cpu()

        num_elements = torch.prod(torch.Tensor(list(predvar.shape[1:])))
        loss_epoch += torch.sum(torch.pow(predvar - outvar, 2)) / num_elements
        num_examples += predvar.shape[0]

        # Plotting
        if i == 0:
            predvar = predvar.numpy()
            outvar = outvar.numpy()
            for chan in channels:
                plt.close("all")
                fig, ax = plt.subplots(
                    3, predvar.shape[1], figsize=(15, predvar.shape[0] * 5)
                )
                for t in range(outvar.shape[1]):
                    ax[0, t].imshow(predvar[0, t, chan])
                    ax[1, t].imshow(outvar[0, t, chan])
                    ax[2, t].imshow(predvar[0, t, chan] - outvar[0, t, chan])

                fig.savefig(f"era5_validation_channel{chan}_epoch{epoch}.png")

    model.train()
    return loss_epoch / num_examples


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    initialize_mlflow(
        experiment_name="Modulus-Launch-Dev",
        experiment_desc="Modulus launch development",
        run_name="FCN-Training",
        run_desc="FCN ERA5 Training",
        user_name="Modulus User",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=True)  # Modulus launch logger
    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger

    # training datapipe
    datapipe = ERA5HDF5Datapipe(
        data_dir=cfg.training.data_dir,
        stats_dir=cfg.stats_dir,
        channels=[i for i in range(cfg.num_input_channels)],
        num_steps=cfg.training.num_steps,
        num_samples_per_year=cfg.training.num_samples_per_year,  # Need better shard fix
        batch_size=cfg.training.batch_size,
        patch_size=cfg.patch_size,
        num_workers=cfg.num_workers,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
    )
    logger.success(f"Loaded datapipe of size {len(datapipe)}")

    # validation datapipe
    if dist.rank == 0:
        logger.file_logging()
        validation_datapipe = ERA5HDF5Datapipe(
            data_dir=cfg.validation.data_dir,
            stats_dir=cfg.stats_dir,
            channels=[i for i in range(cfg.num_input_channels)],
            num_steps=cfg.validation.num_steps,
            num_samples_per_year=cfg.validation.num_samples_per_year,
            batch_size=cfg.validation.batch_size,
            patch_size=cfg.patch_size,
            device=dist.device,
            num_workers=cfg.num_workers,
            shuffle=False,
        )
        logger.success(f"Loaded validation datapipe of size {len(validation_datapipe)}")

    # instantiate the model ( TODO: rethink this)
    if cfg.model_name == "afno":
        model = AFNO(**cfg.model).to(dist.device)
    elif cfg.model_name == "sfno":
        model = SFNO(**cfg.model).to(dist.device)

    if dist.rank == 0 and wandb.run is not None:
        wandb.watch(
            model, log="all", log_freq=1000, log_graph=(True)
        )  # currently does not work with scripted modules. This will be fixed in the next release of W&B SDK.

    # Distributed data parallel
    if dist.world_size > 1:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # Initialize optimizer and scheduler
    if cfg.optimizer == "fused_adam":
        if APEX_AVAILABLE:
            optimizer = optimizers.FusedAdam(
                model.parameters(),
                betas=cfg.betas,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
            rank_zero_logger.info("using FusedAdam optimizer")
        else:
            raise ImportError(
                "fused_adam is not available. "
                "Install apex from https://github.com/nvidia/apex"
            )
    elif cfg.optimizer == "fused_mixed_precision_lamb":
        if APEX_AVAILABLE:
            optimizer = optimizers.FusedMixedPrecisionLamb(
                model.parameters(),
                betas=cfg.betas,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                max_grad_norm=cfg.optimizer_max_grad_norm,
            )
            rank_zero_logger.info("using FusedMixedPrecisionLamb optimizer")
        else:
            raise ImportError(
                "fused_mixed_precision_lamb is not available. "
                "Install apex from https://github.com/nvidia/apex"
            )
    elif cfg.optimizer == "fused_lamb":
        if APEX_AVAILABLE:
            optimizer = optimizers.FusedLAMB(
                model.parameters(),
                betas=cfg.betas,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                max_grad_norm=cfg.optimizer_max_grad_norm,
            )
            rank_zero_logger.info("using FusedLamb optimizer")
        else:
            raise ImportError(
                "fused_lamb is not available. "
                "Install apex from https://github.com/nvidia/apex"
            )
    elif cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            betas=cfg.betas,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        rank_zero_logger.info("using Adam optimizer")
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        rank_zero_logger.info("using SGD optimizer")
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} is not supported.")

    if cfg.scheduler == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=5, mode="min"
        )
    elif cfg.scheduler == "CosineAnnealingLR":
        eta_min = cfg.scheduler_min_lr if hasattr(cfg, "scheduler_min_lr") else 0.0
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.scheduler_T_max,
            eta_min=eta_min,
        )
    elif cfg.scheduler == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.learnin_rate,
            total_steps=cfg.scheduler_T_max,
            steps_per_epoch=1,
        )
    else:
        scheduler = None
        rank_zero_logger.warning(
            f"Scheduler {cfg.scheduler} is not supported. No scheduler will be used."
        )

    if cfg.lr_warmup_steps > 0:  # NOTE different from era5_wind
        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=cfg.lr_warmup_start_factor,  # NOTE use a small value instead of 0 to avoid NaN
            end_factor=1.0,
            total_iters=cfg.lr_warmup_steps,
        )
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                warmup_scheduler,
                scheduler,
            ],  # TODO add a third scheduler to support GraphCast finetuning
            milestones=[cfg.lr_warmup_steps],
        )

    # Attempt to load latest checkpoint if one exists
    loaded_epoch = load_checkpoint(
        "./checkpoints",
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=dist.device,
    )

    @StaticCaptureEvaluateNoGrad(model=model, logger=logger, use_graphs=False)
    def eval_step_forward(my_model, invar):
        return my_model(invar)

    @StaticCaptureTraining(model=model, optim=optimizer, logger=logger)
    def train_step_forward(my_model, invar, outvar):
        # Multi-step prediction
        loss = 0
        # Multi-step not supported
        for t in range(outvar.shape[1]):
            outpred = my_model(invar)
            invar = outpred
            loss += loss_func(outpred, outvar[:, t])
        return loss

    # Main training loop
    max_epoch = 80
    for epoch in range(max(1, loaded_epoch + 1), max_epoch + 1):
        # Wrap epoch in launch logger for console / WandB logs
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=len(datapipe), epoch_alert_freq=10
        ) as log:
            # === Training step ===
            for j, data in enumerate(datapipe):
                invar = data[0]["invar"]
                outvar = data[0]["outvar"]
                loss = train_step_forward(model, invar, outvar)

                log.log_minibatch({"loss": loss.detach()})
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        if dist.rank == 0:
            # Wrap validation in launch logger for console / WandB logs
            with LaunchLogger("valid", epoch=epoch) as log:
                # === Validation step ===
                error = validation_step(
                    eval_step_forward, model, validation_datapipe, epoch=epoch
                )
                log.log_epoch({"Validation error": error})

        if dist.world_size > 1:
            torch.distributed.barrier()

        scheduler.step()

        if (epoch % 5 == 0 or epoch == 1) and dist.rank == 0:
            # Use Modulus Launch checkpoint
            save_checkpoint(
                "./checkpoints",
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )

    rank_zero_logger.info("Finished training!")


if __name__ == "__main__":
    main()
