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


def concat_static_features(
    invar: torch.Tensor,
    data: dict,
    step: int = 0,
    update_coszen: bool = False,
    coszen_channel: int = None,
) -> torch.Tensor:
    """
    Concatenate static features based on the keys in the data dictionary to the input tensor.

    Parameters:
    - invar: Tensor to which the static features will be concatenated.
    - data: Dictionary containing various static features.
    - step: Integer specifying the time step for `cos_zenith`.
    - update_coszen: Boolean, if True, will update the `coszen_channel` of `invar` with the `cos_zenith` feature.
    - coszen_channel: Channel index in `invar` to be updated with the `cos_zenith` feature when `update_coszen` is True.

    Returns:
    - Tensor with concatenated static features.
    """

    if update_coszen:
        if "cos_zenith" in data:
            if coszen_channel is None:
                raise ValueError(
                    "coszen_channel must be specified when update_coszen=True."
                )
            invar[:, coszen_channel] = data["cos_zenith"][:, step]
            return invar

    feature_keys = [
        "cos_zenith",
        "latlon",
        "cos_latlon",
        "geopotential",
        "land_sea_mask",
    ]

    # Extract relevant tensors based on feature_keys and concatenate
    tensors_to_concat = [
        data[key][:, step] if key == "cos_zenith" else data[key]
        for key in feature_keys
        if key in data
    ]

    # Concatenate all tensors in the list
    invar = torch.cat((invar, *tensors_to_concat), dim=1)

    return invar
