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

from typing import Optional, List

import numpy as np
import torch
from torch import nn

from modulus.utils.sfno.distributed import comm
from modulus.utils.sfno.distributed.mappings import gather_from_parallel_region

from modulus.experimental.metrics.climate.geometric import (
    GeometricLpLoss,
    GeometricH1Loss,
)


class LossHandler(nn.Module):
    """
    Wrapper class that handles loss computation
    """

    def __init__(
        self,
        pole_mask: bool = False,
        n_future: int = 0,
        inp_shape: List[int] = [720, 1441],
        in_channels: int = 3,
        out_channels: int = 3,
        channel_names: List[str] = ["u10m", "v10m", "t2m"],
        channel_weight_mode: Optional(str) = "auto",
        channel_weights: Optional(List[float]) = None,
        temporal_std_weighting: bool = False,
        global_stds: Optional[np.ndarray] = None,
        time_diff_stds: Optional[np.ndarray] = None,
        loss_type: str = "geometric-l2",
        absolute: bool = False,
        squared: bool = True,
    ):  # pragma: no cover

        super(LossHandler, self).__init__()

        self.rank = comm.get_rank("matmul")
        self.inp_shape = inp_shape

        pole_mask = 1 if pole_mask else 0
        # TODO (era5_wind): allow for crop offset, otherwise the weighting will not be correct

        if channel_weight_mode == "auto":
            channel_weights = torch.ones(out_channels, dtype=torch.float32)
            for c, chn in enumerate(channel_names):
                if chn in [
                    "u10m",
                    "v10m",
                    "u100m",
                    "v100m",
                    "sp",
                    "msl",
                    "tcwv",
                ]:  # TODO (mnabian): remove hardcoding
                    channel_weights[c] = 0.1
                elif chn in ["t2m", "2d"]:
                    channel_weights[c] = 1.0
                elif chn[0] in ["z", "u", "v", "t", "r", "q"]:
                    pressure_level = float(chn[1:])
                    channel_weights[c] = 0.001 * pressure_level
                else:
                    channel_weights[c] = 0.01
        elif channel_weight_mode == "manual":
            if channel_weights is None:
                raise ValueError(
                    "Channel weights must be provided when using manual weighting"
                )
            channel_weights = torch.Tensor(channel_weights).float()
        elif channel_weight_mode is None:
            channel_weights = torch.ones(out_channels, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown channel weight mode {channel_weight_mode}")

        # renormalize the weights to one
        channel_weights = channel_weights.reshape(1, -1, 1, 1)
        channel_weights = channel_weights / torch.sum(channel_weights)

        # add temporal std weighting
        if temporal_std_weighting:
            if global_stds is None or time_diff_stds is None:
                raise ValueError(
                    "Global and time difference standard deviations must be provided when using temporal weighting"
                )
            eps = 1e-6
            global_stds = global_stds.reshape(1, -1, 1, 1)[:, in_channels]
            time_diff_stds = time_diff_stds.reshape(1, -1, 1, 1)[:, in_channels]
            time_var_weights = global_stds / (time_diff_stds + eps)
            if squared:
                time_var_weights = time_var_weights**2
            channel_weights = channel_weights * time_var_weights

        self.channel_weights = channel_weights
        self.register_buffer(
            "channel_weights", channel_weights
        )  # TODO (mnabian): check if this works with model packaging

        if loss_type == "l2":
            self.loss_obj = GeometricLpLoss(
                inp_shape,
                p=2,
                absolute=absolute,
                pole_mask=pole_mask,
                jacobian="flat",
            )
        elif loss_type == "l1":
            self.loss_obj = GeometricLpLoss(
                inp_shape,
                p=1,
                absolute=absolute,
                pole_mask=pole_mask,
                jacobian="flat",
            )
        elif loss_type == "geometric-l2":
            self.loss_obj = GeometricLpLoss(
                inp_shape,
                p=2,
                absolute=absolute,
                squared=squared,
                pole_mask=pole_mask,
            )
        elif loss_type == "geometric-l1":
            self.loss_obj = GeometricLpLoss(
                inp_shape, p=1, absolute=absolute, pole_mask=pole_mask
            )
        elif loss_type == "geometric-h1":
            self.loss_obj = GeometricH1Loss(
                inp_shape, absolute=absolute, squared=squared
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")

        # weighting factor for the case of multistep training
        # TODO (era5_wind) change hardcoded weighting
        multistep_weight = torch.arange(1, n_future + 2, dtype=torch.float32)
        multistep_weight = multistep_weight / torch.sum(multistep_weight)
        multistep_weight = multistep_weight.reshape(-1, 1, 1, 1)
        self.register_buffer(
            "multistep_weight", multistep_weight
        )  # TODO (mnabian): check if this works with model packaging

        # decide whether to gather the input
        self.do_gather_input = False
        if comm.get_size("h") * comm.get_size("w") > 1:
            self.do_gather_input = True

    @torch.jit.ignore
    def _gather_input(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        # combine data
        # h
        xh = gather_from_parallel_region(x, -2, "h")
        xw = gather_from_parallel_region(xh, -1, "w")

        # crop
        x = xw[:, :, : self.inp_shape[0], : self.inp_shape[1]]

        return x

    def is_distributed(
        self,
    ):  # pragma: no cover  # TODO (mnabian): check if this is needed
        """Returns whether the loss is distributed or not (always False)"""
        return False

    def forward(
        self, prd: torch.Tensor, tar: torch.Tensor, inp: torch.Tensor
    ):  # pragma: no cover

        if self.do_gather_input:
            prd = self._gather_input(prd)
            tar = self._gather_input(tar)

        if hasattr(self, "minmax"):  # TODO (mnabian): check where this is assigned
            channel_weights = torch.ones_like(self.channel_weights)
            channel_weights = channel_weights / torch.sum(channel_weights)
            channel_weights += self.channel_weights.abs() / torch.sum(
                self.channel_weights.abs()
            )
        else:
            channel_weights = self.channel_weights

        if self.training:
            channel_weights = (channel_weights * self.multistep_weight).reshape(
                1, -1, 1, 1
            )

        return self.loss_obj(prd, tar, channel_weights)
