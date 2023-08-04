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
from omegaconf import DictConfig

from modulus.models.afno import AFNO
from modulus.experimental.models.sfno.sfnonet import (
    SphericalFourierNeuralOperatorNet as SFNO,
)
from modulus.datapipes.climate import ERA5HDF5Datapipe
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.experimental.metrics.general.lp_error import lp_error

from modulus.launch.logging import (
    LaunchLogger,
    PythonLogger,
    RankZeroLoggingWrapper,
    initialize_mlflow,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint

from loss_handler import LossHandler
from optimizer_handler import OptimizerHandler

try:  # TODO (mnabian): better handle this
    from apex import optimizers
    apex_available = True
except ImportError:
    apex_available = False


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
        #stats_dir=cfg.stats_dir,  # TODO (mnabian): uncomment
        channels=[i for i in range(cfg.model.in_channels)],
        num_steps=cfg.training.num_steps,
        num_samples_per_year=cfg.training.num_samples_per_year,  # Need better shard fix
        batch_size=cfg.training.batch_size,
        patch_size=cfg.model.patch_size,
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
            #stats_dir=cfg.stats_dir,   # TODO (mnabian): uncomment
            channels=[i for i in range(cfg.model.in_channels)],
            num_steps=cfg.validation.num_steps,
            num_samples_per_year=cfg.validation.num_samples_per_year,
            batch_size=cfg.validation.batch_size,
            patch_size=cfg.model.patch_size,
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
    opt_handler = OptimizerHandler(model.parameters(), cfg, apex_available, rank_zero_logger)
    optimizer = opt_handler.get_optimizer()
    scheduler = opt_handler.get_scheduler()

    # Initialize loss function
    if cfg.loss_type == "relative_l2_error":  # TODO (mnabian): move to loss handler
        loss_func = partial(lp_error, p=2, relative=True, reduce=True)
    elif cfg.loss_type == "l2_error":
        loss_func = partial(lp_error, p=2, relative=False, reduce=True)
    else:
        loss_func = LossHandler(
            pole_mask=cfg.pole_mask,
            n_future=cfg.n_future,
            inp_shape=cfg.inp_shape,
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            channel_names=cfg.channel_names,
            channel_weight_mode=cfg.channel_weight_mode,
            channel_weights=cfg.channel_weights,
            temporal_std_weighting=cfg.temporal_std_weighting,
            global_stds=cfg.global_stds,  # TODO read from npz
            time_diff_stds=cfg.time_diff_stds,  # TODO read from npz
            loss_type=cfg.loss_type,
            absolute=cfg.absolute,
            squared=cfg.squared,
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
    for epoch in range(max(1, loaded_epoch + 1), cfg.max_epoch + 1):
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
