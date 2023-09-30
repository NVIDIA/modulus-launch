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

# Third-party library imports
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

# PyTorch and related library imports
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR

# Modulus specific imports for distributed management, data, logging, checkpointing
from modulus.distributed import DistributedManager
from modulus.experimental.datapipes.climate import ClimateHDF5Datapipe
from modulus.launch.logging import (
    LaunchLogger,
    PythonLogger,
    RankZeroLoggingWrapper,
    initialize_mlflow,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.models import Module
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad

# Local imports for additional utilities and validation
from data_helpers import concat_static_features
from utils.validation import validation_step


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # Initializing distributed environment for parallel training
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initializing logging tools for tracking experiments
    initialize_mlflow(
        experiment_name="Modulus-Launch-Dev",
        experiment_desc="Modulus launch development",
        run_name=f"FCN-Training-{cfg.model_name}",
        run_desc="FCN ERA5 Training",
        user_name="Modulus User",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=True)  # Modulus launch logger
    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger

    # Setting up the training data loader
    datapipe = ClimateHDF5Datapipe(
        data_dir=cfg.training.data_dir,
        # stats_dir=cfg.stats_dir,  # TODO (mnabian): uncomment
        channels=[
            i for i in range(cfg.model.in_channels - cfg.num_static_channels)
        ],  # TODO (mnabian): check this
        batch_size=cfg.training.batch_size,
        stride=cfg.stride,
        dt=cfg.dt,
        start_year=cfg.training.start_year,
        num_steps=cfg.training.num_steps,
        lsm_filename=to_absolute_path(cfg.lsm_path),
        geopotential_filename=to_absolute_path(cfg.geopotential_path),
        use_latlon=cfg.use_latlon,
        use_cos_zenith=cfg.use_cos_zenith,
        patch_size=cfg.patch_size,
        num_samples_per_year=cfg.training.num_samples_per_year,
        shuffle=True,
        num_workers=cfg.num_workers,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
    )
    logger.success(f"Loaded datapipe of size {len(datapipe)}")

    # Setting up the validation data loader
    logger.file_logging()
    validation_datapipe = ClimateHDF5Datapipe(
        data_dir=cfg.validation.data_dir,
        # stats_dir=cfg.stats_dir,   # TODO (mnabian): uncomment
        channels=[i for i in range(cfg.model.in_channels - cfg.num_static_channels)],
        batch_size=cfg.validation.batch_size,
        stride=cfg.stride,
        dt=cfg.dt,
        start_year=cfg.validation.start_year,
        num_steps=cfg.validation.num_steps,
        lsm_filename=to_absolute_path(cfg.lsm_path),
        geopotential_filename=to_absolute_path(cfg.geopotential_path),
        use_latlon=cfg.use_latlon,
        use_cos_zenith=cfg.use_cos_zenith,
        patch_size=cfg.patch_size,
        num_samples_per_year=cfg.validation.num_samples_per_year,
        num_workers=cfg.num_workers,
        shuffle=False,
        device=dist.device,
    )
    logger.success(f"Loaded validation datapipe of size {len(validation_datapipe)}")

    # Model instantiation using the configuration
    model = Module.instantiate(
        {
            "__name__": cfg.model_name,
            "__args__": OmegaConf.to_container(cfg.model, resolve=True),
        }
    )
    model = model.to(dist.device)

    # If in distributed mode, wrap the model for distributed training
    if dist.distributed:
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
        torch.device.synchronize()

    # Setting up the optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=cfg.optimizer.betas,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        fused=cfg.optimizer.fused,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.num_epochs)

    # Loss function instantiation
    loss_function = torch.nn.MSELoss()

    # Load the latest checkpoint if it exists
    loaded_epoch = load_checkpoint(
        cfg.checkpoint_path,
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=dist.device,
    )

    # Evaluation step wrapped with static capture
    @StaticCaptureEvaluateNoGrad(
        model=model, logger=logger, use_graphs=False, use_autocast=cfg.use_amp
    )
    def eval_step_forward(my_model, invar):
        return my_model(invar)

    # Training step wrapped with static capture
    # This decorator will apply optimizations including AMP and Cuda Graphs
    @StaticCaptureTraining(
        model=model,
        optim=optimizer,
        logger=logger,
        use_graphs=cfg.use_graphs,
        use_autocast=cfg.use_amp,
    )
    def train_step(my_model, invar, outvar):
        # Multi-step prediction
        loss = 0
        for t in range(outvar.shape[1]):
            outpred = my_model(invar)
            invar = outpred
            loss += loss_function(outpred, outvar[:, t])
        return loss

    # Main training loop
    for epoch in range(max(1, loaded_epoch + 1), cfg.max_epoch + 1):

        # Training for one epoch
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(datapipe),
            epoch_alert_freq=cfg.epoch_alert_freq,
        ) as log:
            # training step
            for j, data in enumerate(datapipe):  # [B, T, C, H, W]
                data = data[0]
                invar = data["state_seq"][:, 0]
                invar = concat_static_features(invar, data, step=0)
                outvar = data["state_seq"][:, 1:]
                loss = train_step(model, invar, outvar)
                log.log_minibatch({"loss": loss.detach()})
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # Validation after each epoch
        if dist.rank == 0:
            with LaunchLogger("valid", epoch=epoch) as log:
                # validation step
                error = validation_step(
                    eval_step_forward, model, validation_datapipe, epoch=epoch
                )
                log.log_epoch({"Validation error": error})

        # Barrier to ensure all processes sync up before proceeding
        if dist.world_size > 1:
            torch.distributed.barrier()

        # Update the learning rate scheduler
        scheduler.step()

        # Save checkpoints periodically
        if epoch % cfg.save_freq == 0 or epoch == 1:
            save_checkpoint(
                cfg.checkpoint_path,
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )

    rank_zero_logger.info(f"Finished training the {cfg.model_name} model!")


# Entry point of the script
if __name__ == "__main__":
    main()
