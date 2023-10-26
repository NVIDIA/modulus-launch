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
import datetime
import xarray as xr
import hydra
from omegaconf import DictConfig
import logging

from modulus.datapipes.climate import ERA5Mirror


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Make Mirror to download data
    logging.getLogger().setLevel(logging.ERROR)  # Suppress logging from cdsapi
    mirror = ERA5Mirror(base_path=os.path.join(cfg.base_path, "era5_zarr_data"))

    # Download data
    date_range = (datetime.date(2000, 1, 1), datetime.date(2000, cfg.nr_months, 29))
    hours = [cfg.dt * i for i in range(0, 24 // cfg.dt)]
    zarr_paths = mirror.download(cfg.variables, date_range, hours)

    # Open the zarr files and construct the xarray from them
    zarr_arrays = [xr.open_zarr(path) for path in zarr_paths]
    era5_xarray = xr.concat(
        [z[list(z.data_vars.keys())[0]] for z in zarr_arrays], dim="channel"
    )
    era5_xarray = era5_xarray.transpose("time", "channel", "latitude", "longitude")
    era5_xarray.name = "fields"
    era5_xarray = era5_xarray.astype("float32")

    # Save h5 file to disk
    h5_path = os.path.join(cfg.base_path, "era5_data.h5")
    era5_xarray.to_netcdf(h5_path, engine="h5netcdf")


if __name__ == "__main__":
    main()
