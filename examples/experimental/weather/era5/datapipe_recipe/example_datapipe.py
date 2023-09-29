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

import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import datetime

from modulus.experimental.datapipes.climate import ClimateHDF5Datapipe
from modulus.distributed import DistributedManager

from numpy_zenith_angle import cos_zenith_angle


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    # create data pipe
    dp = ClimateHDF5Datapipe(
        data_dir=cfg.data_dir,
        stats_dir=cfg.stats_dir,
        channels=cfg.channels,
        batch_size=cfg.batch_size,
        stride=cfg.stride,
        dt=cfg.dt,
        start_year=cfg.start_year,
        num_steps=cfg.num_steps,
        lsm_filename=cfg.land_sea_mask_filename,
        geopotential_filename=cfg.geopotential_filename,
        use_cos_zenith=cfg.use_cos_zenith,
        use_latlon=cfg.use_latlon,
        shuffle=True,
    )

    for data in dp:
        for i in range(cfg.num_steps):
            fig, ax = plt.subplots(1, 5, figsize=(20, 4))
            ax[0].imshow(
                data[0]["state_seq"][0, i, 0, :, :].detach().cpu().numpy(),
                origin="lower",
            )
            ax[0].set_title("TAS")
            # ax[1].imshow(data[0]["land_sea_mask"][0, 0, :, :].detach().cpu().numpy(), origin="lower")
            # ax[2].imshow(data[0]["geopotential"][0, 0, :, :].detach().cpu().numpy(), origin="lower")
            ax[1].imshow(
                data[0]["latlon"][0, 0, :, :].detach().cpu().numpy(), origin="lower"
            )
            ax[1].set_title("latlon")
            ax[2].imshow(
                data[0]["cos_latlon"][0, 0, :, :].detach().cpu().numpy(), origin="lower"
            )
            ax[2].set_title("cos_latlon")
            ax[3].imshow(
                data[0]["cos_zenith"][0, i, 0, :, :].detach().cpu().numpy(),
                origin="lower",
            )
            ax[3].set_title("cos_zenith (dali)")

            # get numpy zenith angle
            timestamp = datetime.datetime.fromtimestamp(
                data[0]["timestamps"][0, i].detach().cpu().numpy().astype(int)
            )
            lat = data[0]["latlon"][0, 0, :, :].detach().cpu().numpy()
            lon = data[0]["latlon"][0, 1, :, :].detach().cpu().numpy()
            print(
                (
                    1.0 * 3600
                    + data[0]["timestamps"][0, i].detach().cpu().numpy()
                    - 946756800.0
                )
                / (24 * 3600)
                / 36525.0
            )
            cos_zenith = cos_zenith_angle(timestamp, lon, lat)
            ax[4].imshow(cos_zenith, origin="lower")
            ax[4].set_title("cos_zenith (original numpy)")

            plt.show()
            plt.close()


if __name__ == "__main__":
    main()
