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

import os
import gc
import time
import numpy as np
from tqdm import tqdm

# gpu info
import pynvml

# torch
import torch
import torch.cuda.amp as amp

import logging
import wandb

from modulus.models.sfno.preprocessor import get_preprocessor
from models import get_model
from modulus.datapipes.climate.sfno.dataloader import get_dataloader
from modulus.utils.sfno.distributed.mappings import init_gradient_reduction_hooks
from apex import optimizers
from modulus.utils.sfno.loss import LossHandler
from modulus.utils.sfno.metric import MetricsHandler

# distributed computing stuff
from modulus.utils.sfno.distributed import comm
import torch.distributed as dist

# for the manipulation of state dict
from collections import OrderedDict

# visualization utils
import visualize

# for counting model parameters
from helpers import count_parameters

from modulus.launch.logging import (
    PythonLogger,
    LaunchLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
)


class Inferencer:
    # jit stuff
    def _compile_model(self, inp_shape):
        if self.params.jit_mode == "script":
            if dist.is_initialized() and not self.params.disable_ddp:
                self.model.module = torch.jit.script(self.model.module)
            else:
                self.model = torch.jit.script(self.model)
            self.model_train = self.model
            self.model_eval = self.model

        elif self.params.jit_mode == "inductor":
            self.model = torch.compile(self.model)
            self.model_train = self.model
            self.model_eval = self.model

        else:
            self.model_train = self.model
            self.model_eval = self.model

        return

    # graph stuff
    def _capture_model(self, capture_stream, inp_shape, tar_shape, num_warmup_steps=20):
        matmul_comm_size = comm.get_size("matmul")

        # modify inp shape due to model parallelism
        if self.params.split_data_channels:
            inp_shape_eff = (
                inp_shape[0],
                (inp_shape[1] + matmul_comm_size - 1) // matmul_comm_size,
                inp_shape[2],
                inp_shape[3],
            )

            tar_shape_eff = (
                tar_shape[0],
                (tar_shape[1] + matmul_comm_size - 1) // matmul_comm_size,
                tar_shape[2],
                tar_shape[3],
            )
        else:
            inp_shape_eff = (inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3])

            tar_shape_eff = (tar_shape[0], tar_shape[1], tar_shape[2], tar_shape[3])

        # print(inp_shape_eff, tar_shape_eff)
        self.static_inp = torch.zeros(
            inp_shape_eff, dtype=torch.float32, device=self.device
        )
        self.static_tar = torch.zeros(
            tar_shape_eff, dtype=torch.float32, device=self.device
        )

        if self.params.enable_nhwc:
            self.static_inp = self.static_inp.to(memory_format=torch.channels_last)
            self.static_tar = self.static_tar.to(memory_format=torch.channels_last)

        # set to train
        self._set_train()

        # do capture
        if capture_stream is None:
            capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(num_warmup_steps):
                self.model_train.zero_grad(set_to_none=True)

                # FW
                with amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                    self.static_pred = self.model_train(self.static_inp).to(self.device)
                    self.static_loss = self.loss_obj(
                        self.static_pred, self.static_tar, self.static_inp
                    )

                # BW
                self.gscaler.scale(self.static_loss).backward()

            # sync here
            capture_stream.synchronize()

            gc.collect()
            torch.cuda.empty_cache()

            # create graph
            self.graph = torch.cuda.CUDAGraph()

            # zero grads before capture:
            self.model_train.zero_grad(set_to_none=True)

            # start capture
            self.graph.capture_begin()

            # FW
            with amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                self.static_pred = self.model_train(self.static_inp)
                self.static_loss = self.loss_obj(
                    self.static_pred, self.static_tar, self.static_inp
                )

            # BW
            self.gscaler.scale(self.static_loss).backward()

            # end capture
            self.graph.capture_end()

        torch.cuda.current_stream().wait_stream(capture_stream)

        return

    def _get_time_stats(self):
        # get some stats: make data shared with tensor from the class
        out_bias, out_scale = self.valid_dataloader.get_output_normalization()
        mult_cpu = torch.from_numpy(out_scale)[0, :, 0, 0]

        # compute
        if self.params.enable_synthetic_data:
            clim = torch.zeros(
                [
                    self.params.N_out_channels,
                    self.params.img_crop_shape_x,
                    self.params.img_crop_shape_y,
                ],
                dtype=torch.float32,
                device=self.device,
            )

        else:
            # full bias and scale
            in_bias, in_scale = self.valid_dataloader.get_input_normalization()
            in_bias = in_bias[0, ...]
            in_scale = in_scale[0, ...]

            # we need this window
            start_x = self.params.img_crop_offset_x
            end_x = start_x + self.params.img_crop_shape_x
            start_y = self.params.img_crop_offset_y
            end_y = start_y + self.params.img_crop_shape_y

            # now we crop the time means
            time_means = np.load(self.params.time_means_path)[
                0, self.params.out_channels, start_x:end_x, start_y:end_y
            ]
            clim = torch.as_tensor(
                (time_means - in_bias) / in_scale, dtype=torch.float32
            )
        return mult_cpu, clim

    def _update_parameters(self, params):
        """
        This could be moved potentially. The idea is to process params and handle the logics for params
        """

        params.in_channels = self.valid_dataset.in_channels
        params.N_in_channels = len(self.valid_dataset.in_channels)
        params.out_channels = self.valid_dataset.out_channels
        params.N_out_channels = len(self.valid_dataset.out_channels)

        params.img_shape_x = self.valid_dataset.img_shape_x
        params.img_shape_y = self.valid_dataset.img_shape_y

        params.img_crop_shape_x = self.valid_dataset.img_crop_shape_x
        params.img_crop_shape_y = self.valid_dataset.img_crop_shape_y
        params.img_crop_offset_x = self.valid_dataset.img_crop_offset_x
        params.img_crop_offset_y = self.valid_dataset.img_crop_offset_y

        params.img_local_shape_x = self.valid_dataset.img_local_shape_x
        params.img_local_shape_y = self.valid_dataset.img_local_shape_y
        params.img_local_offset_x = self.valid_dataset.img_local_offset_x
        params.img_local_offset_y = self.valid_dataset.img_local_offset_y

        # derived quantities
        params["N_in_predicted_channels"] = params.N_in_channels

        # sanitization:
        if not hasattr(params, "add_zenith"):
            params["add_zenith"] = False

        # input channels
        # zenith channel is appended to all the samples, so we need to do it here
        if params.add_zenith:
            params.N_in_channels += 1

        if params.n_history >= 1:
            params.N_in_channels = (params.n_history + 1) * params.N_in_channels
            params.N_in_predicted_channels *= params.n_history + 1

        # these are static and the same for all samples in the same time history
        if params.add_grid:
            n_grid_chan = 2
            if (params.gridtype == "sinusoidal") and hasattr(
                params, "grid_num_frequencies"
            ):
                n_grid_chan *= params.grid_num_frequencies
            params.N_in_channels += n_grid_chan

        if params.add_orography:
            params.N_in_channels += 1

        if params.add_landmask:
            params.N_in_channels += 2

        # target channels
        params.N_target_channels = (params.n_future + 1) * params.N_out_channels

        # MISC parameters
        if not hasattr(params, "history_normalization_mode"):
            params["history_normalization_mode"] = "none"

        if not hasattr(params, "multigrid_mode"):
            params["multigrid_mode"] = "none"

        if not hasattr(params, "num_visualization_workers"):
            params["num_visualization_workers"] = 1

        if not hasattr(params, "log_video"):
            params["log_video"] = 0

        # automatically detect wind channels and keep track of them
        if hasattr(params, "channel_names") and not hasattr(params, "wind_channels"):
            channel_names = params.channel_names
            channel_dict = {
                channel_names[ch]: ch
                for ch in set(params.in_channels + params.out_channels)
            }
            wind_channels = []
            for chn, ch in channel_dict.items():
                if chn[0] == "u":
                    vchn = "v" + chn[1:]
                    if vchn in channel_dict.keys():
                        # wind_channels.append(ch, channel_dict[vchn])
                        wind_channels = wind_channels + [ch, channel_dict[vchn]]
            params["wind_channels"] = wind_channels

        if not hasattr(params, "load_checkpoint"):
            params["load_checkpoint"] = "legacy"

        return params

    def __del__(self):
        if self.params.log_to_wandb:
            wandb.finish()

    def __init__(self, params, world_rank):
        self.params = None
        self.world_rank = world_rank
        self.rank = world_rank
        self.data_parallel_rank = comm.get_rank("data")
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

        # setup modulus logger
        self.logger = PythonLogger("main")  # General python logger
        # reenable later
        # if self.world_rank == 0:
        #     self.logger.file_logging(file_name=os.path.join(params.experiment_dir, "out.log"))
        self.rank_zero_logger = RankZeroLoggingWrapper(self.logger, self)

        # nvml stuff
        if params.log_to_screen:
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)

        # set amp_parameters
        self.amp_enabled = params.amp_mode != "none"
        self.amp_dtype = (
            torch.float16
            if (params.amp_mode == "fp16")
            else torch.bfloat16
            if (params.amp_mode == "bf16")
            else None
        )

        if params.log_to_wandb:
            # login first:
            wandb.login()
            # init
            wandb.init(
                dir=params.experiment_dir,
                config=params,
                name=params.wandb_name,  # if not params.resuming else None,
                group=params.wandb_group,  # if not params.resuming else None,
                project=params.wandb_project,
                entity=params.wandb_entity,
                resume=params.resuming,
            )

        # data loader
        self.rank_zero_logger.info("initializing data loader")

        # just a dummy dataloader
        self.train_dataloader, self.train_dataset, self.train_sampler = get_dataloader(
            params,
            params.inf_data_path,
            train=True,
            device=self.device,
        )
        self.valid_dataloader, self.valid_dataset = get_dataloader(
            params,
            params.inf_data_path,
            train=False,
            final_eval=True,
            device=self.device,
        )

        self.rank_zero_logger.info("data loader initialized")

        # update params
        params = self._update_parameters(params)

        # save params
        self.params = params

        # init preprocessor and model
        self.model = get_model(params).to(self.device)
        self.preprocessor = self.model.preprocessor

        # define process group for DDP, we might need to override that
        if dist.is_initialized() and not params.disable_ddp:
            ddp_process_group = comm.get_group("data")

        if params.log_to_wandb:
            wandb.watch(self.model)

        # print model
        if self.world_rank == 0:
            print(self.model)

        # metrics handler
        mult_cpu, clim = self._get_time_stats()
        self.metrics = MetricsHandler(self.params, mult_cpu, clim, self.device)
        self.metrics.initialize_buffers()

        # loss handler
        self.loss_obj = LossHandler(self.params, d=2)
        self.loss_obj = self.loss_obj.to(self.device)
        if self.params.enable_nhwc:
            self.loss_obj = self.loss_obj.to(memory_format=torch.channels_last)

        if not params.resuming:
            if params.nettype == "unet":
                self.model.apply(self.model.get_weights_function(params.weight_init))

        self.capturable_optimizer = False
        betas = (params.optimizer_beta1, params.optimizer_beta2)
        if params.optimizer_type == "FusedAdam":
            self.rank_zero_logger.info("using FusedAdam")
            self.optimizer = optimizers.FusedAdam(
                self.model.parameters(),
                betas=betas,
                lr=params.lr,
                weight_decay=params.weight_decay,
            )
        elif params.optimizer_type == "FusedLAMB":
            try:
                import doesnotexist
                from apex.optimizers import FusedMixedPrecisionLamb

                self.rank_zero_logger.info("using FusedMixedPrecisionLamb")
                self.optimizer = FusedMixedPrecisionLamb(
                    self.model.parameters(),
                    betas=betas,
                    lr=params.lr,
                    weight_decay=params.weight_decay,
                    max_grad_norm=params.optimizer_max_grad_norm,
                )
                self.capturable_optimizer = True
            except ImportError:
                self.rank_zero_logger.info("using FusedLAMB")
                self.optimizer = optimizers.FusedLAMB(
                    self.model.parameters(),
                    betas=betas,
                    lr=params.lr,
                    weight_decay=params.weight_decay,
                    max_grad_norm=params.optimizer_max_grad_norm,
                )
        elif params.optimizer_type == "Adam":
            self.rank_zero_logger.info("using Adam")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        elif params.optimizer_type == "SGD":
            self.rank_zero_logger.info("using SGD")
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=params.lr,
                weight_decay=params.weight_decay,
                momentum=0,
            )
        else:
            raise ValueError(f"Unknown optimizer type {params.optimizer_type}")

        if params.scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.2, patience=5, mode="min"
            )
        elif params.scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=params.scheduler_T_max
            )
        else:
            self.scheduler = None
        if params.lr_warmup_steps > 0:
            from utils.warmup_scheduler import WarmupScheduler

            self.scheduler = WarmupScheduler(
                self.scheduler,
                num_warmup_steps=params.lr_warmup_steps,
                start_lr=params.lr_start,
            )

        self.gscaler = amp.GradScaler(enabled=(self.amp_dtype == torch.float16))

        # we need this further down
        capture_stream = None
        if dist.is_initialized() and not params.disable_ddp:
            capture_stream = torch.cuda.Stream()
            parameter_size_mb = (
                count_parameters(self.model, self.device) * 4 / float(1024 * 1024)
            )
            reduction_size_mb = int(
                (parameter_size_mb / params.parameters_reduction_buffer_count) * 1.05
            )
            with torch.cuda.stream(capture_stream):
                self.model = init_gradient_reduction_hooks(
                    self.model,
                    device_ids=[self.device.index],
                    output_device=[self.device.index],
                    bucket_cap_mb=reduction_size_mb,
                    broadcast_buffers=True,
                    find_unused_parameters=False,
                    gradient_as_bucket_view=True,
                    static_graph=params.checkpointing > 0,
                )
                capture_stream.synchronize()

                # we need to set up some additional gradient reductions
                # if params.model_parallel_size > 1:
                #    init_additional_parameters_reductions(self.model)

            # capture stream sync
            capture_stream.synchronize()

        # lets get one sample from the dataloader:
        # get sample and map to gpu
        iterator = iter(self.train_dataloader)
        data = next(iterator)
        gdata = map(lambda x: x.to(self.device, dtype=torch.float32), data)
        # extract unpredicted features
        inp, tar = self.preprocessor.cache_unpredicted_features(*gdata)
        # flatten
        inp = self.preprocessor.flatten_history(inp)
        tar = self.preprocessor.flatten_history(tar)
        # get shapes
        inp_shape = inp.shape
        tar_shape = tar.shape

        self._compile_model(inp_shape)
        if not self.loss_obj.is_distributed():
            self.loss_obj = torch.jit.script(self.loss_obj)

        # graph capture
        self.graph = None
        if params.cuda_graph_mode != "none":
            self._capture_model(
                capture_stream, inp_shape, tar_shape, num_warmup_steps=20
            )

        # reload checkpoints
        self.iters = 0
        self.startEpoch = 0
        assert (
            (params.pretrained_checkpoint_path is not None),
            "Error, please specify a valid pretrained checkpoint path",
        )
        self.restore_checkpoint(
            params.pretrained_checkpoint_path,
            checkpoint_mode=params["load_checkpoint"],
        )
        self.epoch = self.startEpoch

        if params.log_to_screen:
            pcount = count_parameters(self.model, self.device)
            self.rank_zero_logger.info("Number of trainable model parameters: {pcount}")

    def inference(self):
        # log parameters
        if self.params.log_to_screen:
            # log memory usage so far
            all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / (
                1024.0 * 1024.0 * 1024.0
            )
            max_mem_gb = torch.cuda.max_memory_allocated(device=self.device) / (
                1024.0 * 1024.0 * 1024.0
            )
            self.rank_zero_logger.info(
                f"Scaffolding memory high watermark: {all_mem_gb} GB ({max_mem_gb} GB for pytorch)"
            )
            # announce training start
            self.rank_zero_logger.info("Starting Training Loop...")

        # perform a barrier here to make sure everybody is ready
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        try:
            torch.cuda.reset_peak_memory_stats(self.device)
        except ValueError:
            pass

        training_start = time.time()
        best_valid_loss = 1.0e6

        epoch = 0

        # start timer
        epoch_start = time.time()

        inf_time, inf_logs = self.inference_one_epoch(epoch)

        # end timer
        epoch_end = time.time()

        # create timing logs:

        # training done
        training_end = time.time()
        self.rank_zero_logger.info(
            f"Total training time is {(training_end - training_start):.2f} sec"
        )

        return

    def _set_train(self):
        self.model.train()
        self.loss_obj.train()

    def _set_eval(self):
        self.model.eval()
        self.loss_obj.eval()

    def inference_one_epoch(self, epoch):
        # set to eval
        self._set_eval()

        # clear cache
        torch.cuda.empty_cache()

        # initialize metrics buffers
        self.metrics.zero_buffers()

        # start the timer
        valid_start = time.time()

        with torch.inference_mode():
            with torch.no_grad():
                eval_steps = 0
                for data in tqdm(
                    self.valid_dataloader,
                    desc="Inference progress",
                    disable=not self.params.log_to_screen,
                ):
                    eval_steps += 1

                    # map to gpu
                    gdata = map(lambda x: x.to(self.device, dtype=torch.float32), data)

                    # preprocess
                    inp, tar = self.preprocessor.cache_unpredicted_features(*gdata)
                    inp = self.preprocessor.flatten_history(inp)

                    # split list of targets
                    tarlist = torch.split(tar, 1, dim=1)

                    # do autoregression
                    inpt = inp

                    for idt, targ in enumerate(tarlist):
                        # flatten history of the target
                        targ = self.preprocessor.flatten_history(targ)

                        # FW pass
                        with amp.autocast(
                            enabled=self.amp_enabled, dtype=self.amp_dtype
                        ):
                            pred = self.model_eval(inpt)
                            loss = self.loss_obj(pred, targ, inpt)

                        # put in the metrics handler
                        self.metrics.update(pred, targ, loss, idt)

                        # append history
                        inpt = self.preprocessor.append_history(inpt, pred, idt)

        # create final logs
        logs, acc_curve = self.metrics.finalize(final_inference=True)

        # save the acc curve

        if self.world_rank == 0:
            np.save(
                os.path.join(self.params.experiment_dir, "acc_curve.npy"),
                acc_curve.cpu().numpy(),
            )

            if self.params.ifs_acc_path is not None:
                visualize.plot_ifs_acc_comparison(acc_curve, self.params, self.epoch)

        # global sync is in order
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        # timer
        inference_time = time.time() - valid_start

        return inference_time, logs

    def test_model_output(self, model):
        """helper to test checkpointing"""
        inp_shape = (
            self.params.batch_size,
            self.params.N_in_channels,
            self.params.img_shape_local_x,
            self.params.img_shape_local_y,
        )
        matmul_comm_size = comm.get_size("matmul")

        # modify inp shape due to model parallelism
        if self.params.split_data_channels:
            inp_shape_eff = (
                inp_shape[0],
                (inp_shape[1] + matmul_comm_size - 1) // matmul_comm_size,
                inp_shape[2],
                inp_shape[3],
            )
        else:
            inp_shape_eff = (inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3])

        random_tensor = os.path.join(
            self.params.experiment_dir,
            "random_tensor{}.npy".format(comm.get_rank("model")),
        )
        if not os.path.exists(random_tensor):
            y = torch.rand(inp_shape_eff, dtype=torch.float).cpu().numpy()
            np.save(random_tensor, y)

        y = torch.from_numpy(np.load(random_tensor)).type(torch.float).to(self.device)
        out = model(y).detach().cpu().numpy()
        random_output = os.path.join(
            self.params.experiment_dir,
            "random_output{}.npy".format(comm.get_rank("model")),
        )
        if os.path.exists(random_output):
            out_old = np.load(random_output)
            diff = (out - out_old).flatten()
            self.rank_zero_logger.info(
                "Diff metrics: norm = {}, max = {}, min = {}".format(
                    np.linalg.norm(diff), np.max(diff), np.min(diff)
                )
            )
        np.save(random_output, out)

    def restore_checkpoint(self, checkpoint_path, checkpoint_mode="flexible"):
        """We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function"""

        # legacy mode
        if checkpoint_mode == "legacy":
            checkpoint_fname = checkpoint_path.format(mp_rank=comm.get_rank("model"))
            self.rank_zero_logger.info(
                "Loading checkpoint {checkpoint_fname} in legacy mode"
            )
            checkpoint = torch.load(
                checkpoint_fname, map_location="cuda:{}".format(self.device.index)
            )

            # this is reworked to avoid loading modules related to the SHT
            state_dict = checkpoint["model_state"]
            # a hacky workaround to remove SHT params from state dict
            if True:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    sht_strings = [
                        "forward_transform",
                        "inverse_transform",
                        "sht",
                        "isht",
                        "sht_down",
                        "isht_up",
                        ".ii",
                        ".jj",
                        ".pct",
                        "trans_down",
                        "itrans_up",
                        "trans",
                        "itrans",
                    ]
                    contains = [string in k for string in sht_strings]
                    if True not in contains:
                        # to be able to deal with older implementations we need to reshape any weights from norm layers
                        # this can be removed in the future
                        if "norm" in k:
                            v = v.reshape(-1)

                        new_state_dict[k] = v

                state_dict = new_state_dict

            self.model.load_state_dict(state_dict, strict=False)

            # we load the dict a second time for the cases in which the previous run was not conducted with multistep
            if self.params.n_future > 0:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = "module." + "model." + k[7:]
                    new_state_dict[name] = v

                self.model.load_state_dict(new_state_dict, strict=False)

        # new flexible mode allows to load models in arbitrary model-parallel configurations
        elif checkpoint_mode == "flexible":
            # when loading the weights in flexble mode we exclusively use mp_rank=0 and load them onto the cpu
            checkpoint_fname = checkpoint_path.format(mp_rank=0)
            self.rank_zero_logger.info(
                "Loading checkpoint {checkpoint_fname} in flexible mode"
            )
            checkpoint = torch.load(
                checkpoint_fname, map_location="cuda:{}".format(self.device.index)
            )

            # this is reworked to avoid loading modules related to the SHT
            state_dict = checkpoint["model_state"]
            new_state_dict = OrderedDict()

            for k, v in self.model.named_parameters():
                if k in state_dict.keys():
                    if hasattr(v, "sharded_dims_mp"):
                        weight_shape = state_dict[k].shape
                        read_ranges = []
                        for d, group in enumerate(v.sharded_dims_mp):
                            # compute the read range for this model
                            if group is None:
                                # read_range = None
                                read_range = slice(0, v.shape[d], 1)
                            else:
                                weight_shape_dist = (
                                    (weight_shape[0] + comm.get_size(group) - 1)
                                ) // comm.get_size(group)
                                read_range = slice(
                                    weight_shape_dist * comm.get_rank(group),
                                    weight_shape_dist * comm.get_rank(group)
                                    + v.shape[d],
                                    1,
                                )

                            read_ranges.append(read_range)

                        new_state_dict[k] = state_dict[k][read_ranges]
                    else:
                        new_state_dict[k] = state_dict[k]

                    # to be able to deal with older implementations we need to reshape any weights from norm layers
                    # this can be removed in the future
                    if "norm" in k:
                        new_state_dict[k] = new_state_dict[k].reshape(-1)

                else:
                    # put a warning here
                    print(f"missing {k}")

            state_dict = new_state_dict
            self.model.load_state_dict(state_dict, strict=False)

        else:
            raise ValueError(f"Unknown checkoint mode {checkpoint_mode}.")
