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

# Get MPI communicator
import mpi4py.MPI
import os

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import os
import datetime
import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
import h5py

from experiment import Experiment
from fancy_bar_plot import fancy_bar_plot

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Make experiments
    experiments = []
    for experiment in cfg.experiments:
        experiments.append(Experiment(base_path=cfg.base_path, **experiment))

    # Run experiments
    if rank == 0:

        # Load era5 data
        try:
            era5_h5 = h5py.File(os.path.join(cfg.base_path, "era5_data.h5"), "r")
        except FileNotFoundError:
            raise FileNotFoundError(
                "Era5 data not found. Please download it with `python fetch_era5.py`"
            )

        # Save different experiments
        for experiment in experiments:
            experiment.save_data(era5_h5["fields"])

    # Wait for rank 0 to finish saving data
    comm.Barrier()

    if rank == 0:
        # Plot compression ratios
        cp_ratios = [experiment.get_compression_ratio() for experiment in experiments]
        cp_ratio_labels = [experiment.plot_name for experiment in experiments]
        fancy_bar_plot(
            cp_ratios,
            cp_ratio_labels,
            "Compression ratios",
            "Compression ratio",
            "Experiment",
            os.path.join(cfg.base_path, "compression_ratios.png"),
        )

        # Plot decompression times
        decomp_slice = [slice(0, s) for s in cfg.decompression_slice]
        decomp_times = [
            experiment.get_decompression_time(decomp_slice)
            for experiment in experiments
            if experiment.device == "gpu"
        ]
        decomp_time_labels = [
            experiment.plot_name
            for experiment in experiments
            if experiment.device == "gpu"
        ]
        fancy_bar_plot(
            decomp_times,
            decomp_time_labels,
            "Decompression times",
            "Decompression GB/s",
            "Experiment",
            os.path.join(cfg.base_path, "decompression_times.png"),
        )

    # Plot throughput times (run on all ranks)
    throughput_slice = [slice(0, s) for s in cfg.throughput_slice]
    throughputs = [
        experiment.get_throughput(throughput_slice, comm) for experiment in experiments
    ]
    if rank == 0:
        throughput_labels = [experiment.plot_name for experiment in experiments]
        fancy_bar_plot(
            throughputs,
            throughput_labels,
            f"Throughput times, MPI size: {size}",
            "Throughput GB/s",
            "Experiment",
            os.path.join(cfg.base_path, "throughput_times.png"),
        )


if __name__ == "__main__":
    main()
