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

# Standard libraries
from functools import partial

# Third-party libraries
import hydra
from omegaconf import DictConfig

# Torch related
import torch
from torch.nn.parallel import DistributedDataParallel

# Modulus imports
from modulus.distributed import DistributedManager
from modulus.experimental.datapipes.climate import ClimateHDF5Datapipe
from modulus.experimental.metrics.general.lp_error import lp_error
from modulus.launch.config import to_absolute_path
from modulus.launch.logging import (
    LaunchLogger,
    PythonLogger,
    RankZeroLoggingWrapper,
    initialize_mlflow,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.models import Module
from modulus.registry import ModelRegistry
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad

# Local imports
from loss_handler import LossHandler
from optimizer_handler import OptimizerHandler
from validation import validation_step
from data_helpers import concat_static_features


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize mlflow and logging
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

    # instantiate the training datapipe
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
        lsm_filename=to_absolute_path(cfg.lsm_filename),
        geopotential_filename=to_absolute_path(cfg.geopotential_filename),
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

    # instantiate the validation datapipe
    if dist.rank == 0:
        logger.file_logging()
        validation_datapipe = ClimateHDF5Datapipe(
            data_dir=cfg.validation.data_dir,
            # stats_dir=cfg.stats_dir,   # TODO (mnabian): uncomment
            channels=[
                i for i in range(cfg.model.in_channels - cfg.num_static_channels)
            ],
            batch_size=cfg.validation.batch_size,
            stride=cfg.stride,
            dt=cfg.dt,
            start_year=cfg.validation.start_year,
            num_steps=cfg.validation.num_steps,
            lsm_filename=to_absolute_path(cfg.lsm_filename),
            geopotential_filename=to_absolute_path(cfg.geopotential_filename),
            use_latlon=cfg.use_latlon,
            use_cos_zenith=cfg.use_cos_zenith,
            patch_size=cfg.patch_size,
            num_samples_per_year=cfg.validation.num_samples_per_year,
            num_workers=cfg.num_workers,
            shuffle=False,
            device=dist.device,
        )
        logger.success(f"Loaded validation datapipe of size {len(validation_datapipe)}")

    # instantiate the model
    registry = ModelRegistry()
    assert (
        cfg.model_name in registry.list_models()
    ), f"Model {cfg.model_name} not found in Modulus registry"
    model = Module.instantiate({"__name__": cfg.model_name, "__args__": cfg.model})
    model = model.to(dist.device)

    # Wrap for distributed data parallel
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

    # Instantiate optimizer and scheduler
    opt_handler = OptimizerHandler(model.parameters(), cfg, rank_zero_logger)
    optimizer = opt_handler.get_optimizer()
    scheduler = opt_handler.get_scheduler()

    # Instantiate loss function
    if cfg.loss_type == "relative_l2_error":  # TODO (mnabian): move to loss handler
        loss_function = partial(lp_error, p=2, relative=True, reduce=True)
    elif cfg.loss_type == "l2_error":
        loss_function = partial(lp_error, p=2, relative=False, reduce=True)
    else:
        loss_function = LossHandler(
            pole_mask=cfg.pole_mask,
            n_future=cfg.n_future,
            inp_shape=cfg.inp_shape,
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            channel_names=cfg.channel_names,
            channel_weight_mode=cfg.channel_weight_mode,
            channel_weights=cfg.channel_weights,
            temporal_std_weighting=cfg.temporal_std_weighting,
            global_stds_path=cfg.global_stds_path,
            time_diff_stds_path=cfg.time_diff_stds_path,
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

    # Evaluation step wrapped with static capture
    @StaticCaptureEvaluateNoGrad(model=model, logger=logger, use_graphs=False)
    def eval_step_forward(my_model, invar):
        return my_model(invar)

    # Training step wrapped with static capture
    @StaticCaptureTraining(model=model, optim=optimizer, logger=logger)
    def train_step_forward(my_model, invar, outvar):
        # Multi-step prediction
        loss = 0
        # Multi-step not supported
        for t in range(outvar.shape[1]):
            outpred = my_model(invar)
            invar = outpred
            loss += loss_function(outpred, outvar[:, t])
        return loss

    # Main training loop
    for epoch in range(max(1, loaded_epoch + 1), cfg.max_epoch + 1):

        # Wrap epoch in launch logger for console / WandB logs
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=len(datapipe), epoch_alert_freq=10
        ) as log:
            # training step
            for j, data in enumerate(datapipe):  # [B, T, C, H, W]
                data = data[0]
                invar = data["state_seq"][:, 0]
                invar = concat_static_features(invar, data, step=0)
                outvar = data["state_seq"][:, 1:]
                loss = train_step_forward(model, invar, outvar)
                log.log_minibatch({"loss": loss.detach()})
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # Wrap validation in launch logger for console / WandB logs
        if dist.rank == 0:
            with LaunchLogger("valid", epoch=epoch) as log:
                # validation step
                error = validation_step(
                    eval_step_forward, model, validation_datapipe, epoch=epoch
                )
                log.log_epoch({"Validation error": error})

        if dist.world_size > 1:
            torch.distributed.barrier()

        scheduler.step()

        # Use Modulus Launch checkpoint
        if (epoch % 5 == 0 or epoch == 1) and dist.rank == 0:
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
