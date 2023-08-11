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
    invar, data, step=0, update_coszen=False, coszen_channel=None
):
    keys = data.keys()
    if update_coszen:
        if "cos_zenith" in keys:
            assert (
                coszen_channel is not None
            ), "coszen_channel must be specified for update_coszen=True"
            invar[:, coszen_channel] = data["cos_zenith"][:, step]
        return invar

    if "cos_zenith" in keys:
        invar = torch.cat((invar, data["cos_zenith"][:, step]), dim=1)
    if "latlon" in keys:
        invar = torch.cat((invar, data["latlon"]), dim=1)
    if "cos_latlon" in keys:
        invar = torch.cat((invar, data["cos_latlon"]), dim=1)
    if "geopotential" in keys:
        invar = torch.cat((invar, data["geopotential"]), dim=1)
    if "land_sea_mask" in keys:
        invar = torch.cat((invar, data["land_sea_mask"]), dim=1)
    return invar
