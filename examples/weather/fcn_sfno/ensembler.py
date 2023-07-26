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
import time
import copy
import numpy as np
from tqdm import tqdm

# gpu info
import pynvml

# torch
import torch
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp

import logging
import wandb

from models import get_model
from modulus.datapipes.climate.sfno.dataloader import get_dataloader
from modulus.utils.sfno.distributed.mappings import gather_from_parallel_region
from modulus.utils.sfno.loss import LossHandler
from modulus.utils.sfno.metric import MetricsHandler

# distributed computing stuff
from modulus.utils.sfno.distributed import comm
import torch.distributed as dist

# for the manipulation of state dict
from collections import OrderedDict

# for counting model parameters
from helpers import count_parameters


class Ensembler:
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
                0,
                self.params.out_channels,
                start_x:end_x,
                start_y:end_y,
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

        params.n_future = 0

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
        self.data_parallel_rank = comm.get_rank("data")
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

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
        if params.log_to_screen:
            logging.info("initializing data loader")

        # we set the number of validation steps manually to 0 so we can abuse the dataloader and load examples without shuffling
        params.valid_autoreg_steps = 0

        # just a dummy dataloader
        self.valid_dataloader, self.valid_dataset = get_dataloader(
            params,
            params.inf_data_path,
            train=False,
            final_eval=True,
            device=self.device,
        )

        if params.log_to_screen:
            logging.info("data loader initialized")

        # update params
        params = self._update_parameters(params)

        # save params
        self.params = params

        # init preprocessor and model
        self.model = get_model(params).to(self.device)
        self.preprocessor = self.model.preprocessor

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

        self.model_eval = self.model

        # reload checkpoints
        self.iters = 0
        self.startEpoch = 0
        assert (
            params.pretrained_checkpoint_path is not None
        ), "Error, please specify a valid pretrained checkpoint path"
        self.restore_checkpoint(
            params.pretrained_checkpoint_path, checkpoint_mode=params["load_checkpoint"]
        )

        self.epoch = self.startEpoch

        # if params.log_to_screen:
        #   logging.info(self.model)
        # counting runs a reduction so we need to count on all ranks before printing on rank 0
        pcount = count_parameters(self.model, self.device)
        if params.log_to_screen:
            logging.info("Number of trainable model parameters: {}".format(pcount))

    def ensemble(self):
        # log parameters
        if self.params.log_to_screen:
            # log memory usage so far
            all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / (
                1024.0 * 1024.0 * 1024.0
            )
            max_mem_gb = torch.cuda.max_memory_allocated(device=self.device) / (
                1024.0 * 1024.0 * 1024.0
            )
            logging.info(
                f"Scaffolding memory high watermark: {all_mem_gb} GB ({max_mem_gb} GB for pytorch)"
            )
            # announce training start
            logging.info("Starting Training Loop...")

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

        ens_time = self.ensemble_one_epoch(epoch)

        # end timer
        epoch_end = time.time()

        # create timing logs:

        # training done
        training_end = time.time()
        if self.params.log_to_screen:
            logging.info(
                "Total training time is {:.2f} sec".format(
                    training_end - training_start
                )
            )

        return

    def _set_train(self):
        self.model.train()
        self.loss_obj.train()

    def _set_eval(self):
        self.model.eval()
        self.loss_obj.eval()

    @torch.jit.ignore
    def _gather_hw(self, x: torch.Tensor) -> torch.Tensor:
        # gather the data over the spatial communicator
        xh = gather_from_parallel_region(x, -2, "h")
        xw = gather_from_parallel_region(xh, -1, "w")
        return xw

    @torch.jit.ignore
    def _gather_data(self, x: torch.Tensor) -> torch.Tensor:
        # gather the data over the spatial communicator
        xd = gather_from_parallel_region(x, -4, "data")
        return xd

    def ensemble_one_epoch(
        self, epoch, log_channels=["u10m", "v10m", "t2m"]
    ):
        # set to eval
        self._set_eval()

        # get channels
        ch = [self.params.channel_names.index(chn) for chn in log_channels]

        # clear cache
        torch.cuda.empty_cache()

        # initialize metrics buffers
        self.metrics.zero_buffers()

        # start the timer
        valid_start = time.time()

        with torch.inference_mode():
            with torch.no_grad():
                # ensemble member list
                enslist = []
                gtslist = []

                # we only use one starting point
                iterator = iter(self.valid_dataloader)

                for ens_step in tqdm(
                    range(self.params.ensemble_members),
                    desc="Ensemble progress",
                    disable=not self.params.log_to_screen,
                ):
                    prdlist = []
                    gtlist = []

                    for idt in range(self.params.ensemble_autoreg_steps):
                        data = next(iterator)
                        gdata = map(
                            lambda x: x.to(self.device, dtype=torch.float32), data
                        )

                        # preprocess
                        inp, tar = self.preprocessor.cache_unpredicted_features(*gdata)
                        inp = self.preprocessor.flatten_history(inp)

                        # split list of targets
                        # tarlist = torch.split(tar, 1, dim=1)

                        if idt == 0:
                            inpt = inp
                            # add the noise if it is the first step
                            # TODO: add more ensembling strategies
                            # TODO: we treat the batch dimension as ensemble dimension so we need to turn off the noise for member 0
                            noise = self.params.noise_amplitude * torch.randn_like(
                                inpt[:, : self.params.N_in_predicted_channels]
                            )
                            inpt[:, : self.params.N_in_predicted_channels] += noise

                        # keep track of gt
                        gt = self._gather_hw(inp).detach().unsqueeze(1).cpu().numpy()
                        gtlist.append(gt[:, :, ch])

                        targ = tar

                        # gather the output and write it out
                        out = self._gather_hw(inpt)
                        # currently disabling data dimension and only working on rank 0
                        # out = self._gather_data(out)
                        out = out.detach().unsqueeze(1).cpu().numpy()
                        prdlist.append(out[:, :, ch])

                        # flatten history of the target
                        targ = self.preprocessor.flatten_history(targ)

                        # FW pass
                        with amp.autocast(
                            enabled=self.amp_enabled, dtype=self.amp_dtype
                        ):
                            pred = self.model_eval(inpt)
                            # loss = self.loss_obj(pred, targ, inpt)

                        # put in the metrics handler
                        # self.metrics.update(pred, targ, loss, idt)

                        # append history, which should also correctlyy append the targets zenith andgle
                        inpt = self.preprocessor.append_history(inpt, pred, idt)

                    ens_member = np.stack(prdlist, axis=1)
                    enslist.append(ens_member)
                    gts_member = np.stack(gtlist, axis=1)
                    gtslist.append(gts_member)

        # we should add a gather over the batch dim probably
        ens_array = np.stack(enslist, axis=0)
        print(ens_array.shape)

        gts_array = np.stack(gtslist, axis=0)
        print(gts_array.shape)

        # # create final logs
        # logs, acc_curve = self.metrics.finalize(final_inference=True)

        # save the acc curve
        if self.world_rank == 0:
            np.save(
                os.path.join(self.params.experiment_dir, "ensemble_output.npy"),
                ens_array,
            )
            np.save(
                os.path.join(self.params.experiment_dir, "gts_output.npy"), gts_array
            )

        # global sync is in order
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        # timer
        inference_time = time.time() - valid_start

        return inference_time

    def restore_checkpoint(self, checkpoint_path, checkpoint_mode="flexible"):
        """We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function"""

        # legacy mode
        if checkpoint_mode == "legacy":
            checkpoint_fname = checkpoint_path.format(mp_rank=comm.get_rank("model"))
            if self.params.log_to_screen:
                logging.info("Loading checkpoint {checkpoint_fname} in legacy mode")
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
            if self.params.log_to_screen:
                logging.info("Loading checkpoint {checkpoint_fname} in flexible mode")
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
                                    weight_shape[0] + comm.get_size(group) - 1
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

        print(torch.sum(self.model.model.pos_embed))
