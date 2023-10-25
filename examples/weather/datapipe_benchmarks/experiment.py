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
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import hydra
from omegaconf import DictConfig
import logging
import numpy as np
import cupy as cp
import h5py
import zarr
import kvikio.zarr
import numcodecs
from numcodecs import Blosc, GZip, Zstd
import mpi4py.MPI as MPI

from zarr_utils import get_chunks_for_slice
from custom_gds import GDSStore

# CPU compressor lookup table
cpu_compressor_lookup = {
    "LZ4": zarr.LZ4(),
    "Gdeflate": GZip(level=1),
    "snappy": Blosc(cname="snappy", clevel=5, shuffle=Blosc.NOSHUFFLE),
    "zstd": Zstd(level=5),
    "none": None,
}

# GPU compressor lookup table
gpu_compressor_lookup = {
    "LZ4": kvikio.zarr.LZ4(),
    "snappy": kvikio.zarr.Snappy(),
    "Gdeflate": kvikio.zarr.Gdeflate(),
    "cascaded": kvikio.zarr.Cascaded(),
    "bitcomp": kvikio.zarr.Bitcomp(),
}


class Experiment:
    def __init__(
        self, base_path, filetype, device, compression_algorithm, batch_codec, chunking
    ):
        self.base_path = base_path
        self.filetype = filetype
        self.device = device
        self.compression_algorithm = compression_algorithm
        self.batch_codec = batch_codec
        self.chunking = chunking
        self.save_path = os.path.join(base_path, self.filename)

    @property
    def filename(self):
        if self.filetype == "hdf5":
            format_str = ".h5"
        elif self.filetype == "zarr":
            format_str = ".zarr"
        return (
            f"array_device_{self.device}_chunking_"
            f"{self.chunking[0]}_{self.chunking[1]}_{self.chunking[2]}_"
            f"{self.chunking[3]}_compression_{self.compression_algorithm}"
            f"_batch_{self.batch_codec}" + format_str
        )

    @property
    def plot_name(self):
        return (
            f"device: {self.device}\n"
            + f"filetype: {self.filetype}\n"
            + f"compression: {self.compression_algorithm}\n"
            + f"batch_codec: {self.batch_codec}\n"
            + f"chunk: {self.chunking}\n"
        )

    def get_codec(self):
        if self.device == "cpu":
            return cpu_compressor_lookup[self.compression_algorithm]
        elif self.device == "gpu" and not self.batch_codec:
            codec = gpu_compressor_lookup[self.compression_algorithm]
            return codec
        elif self.device == "gpu" and self.batch_codec:
            np.set_printoptions(precision=4, suppress=True)
            NVCOMP_CODEC_ID = "nvcomp_batch"
            codec = numcodecs.registry.get_codec(
                dict(id=NVCOMP_CODEC_ID, algorithm=self.compression_algorithm)
            )
            return codec
        else:
            raise ValueError("Invalid device")

    def save_data(self, np_array):
        # Check if file exists
        if os.path.isdir(self.save_path) or os.path.isfile(self.save_path):
            print(f"File {self.save_path} already exists, skipping...")
            return

        # Save hdf5 if filetype is hdf5, otherwise save zarr
        if self.filetype == "hdf5":
            assert (
                self.compression_algorithm == "none"
            ), "HDF5 tests do not support compression"
            with h5py.File(self.save_path, "w") as f:
                # Initialize empty dataset of correct size
                f.create_dataset(
                    "data",
                    shape=np_array.shape,
                    dtype=np.float32,
                    chunks=tuple(self.chunking),
                )
                for i in range(np_array.shape[0]):
                    f["data"][i] = np_array[i]

                del f

        # Save zarr
        elif self.filetype == "zarr":
            # Get Codec
            codec = self.get_codec()

            # Make zarr array
            zarr_array = zarr.array(np_array, chunks=self.chunking, compressor=codec)
            zarr.save_array(GDSStore(self.save_path), zarr_array, compressor=codec)

            del zarr_array

        print(f"Saved {self.save_path}!")

    def open_array(self):
        if self.filetype == "hdf5":
            f = h5py.File(self.save_path, "r")
            return f["data"]
        elif self.filetype == "zarr":
            if self.device == "cpu":
                return zarr.open_array(self.save_path, mode="r")
            elif self.device == "gpu":
                #store = GDSStore(self.save_path)
                store = kvikio.zarr.GDSStore(self.save_path)
                # store = zarr.storage.DirectoryStore(self.save_path)
                # print(self.save_path)
                return zarr.open_array(store, mode="r", meta_array=cp.empty(()))
            else:
                raise ValueError("Invalid device")

    def get_compression_ratio(self):
        # Look at zarr array to get compression ratio
        if self.filetype == "hdf5":
            return 1.0
        elif self.filetype == "zarr":
            zarr_array = self.open_array()
            return zarr_array.nbytes / zarr_array.nbytes_stored

    def get_decompression_time(self, slices, nr_repeats=16):
        # Check compression algorithm
        if self.compression_algorithm == "none":
            raise ValueError("Cannot get decompression time for uncompressed data")

        # Get zarr array (only zarr arrays support decompression time)
        zarr_array = self.open_array()

        # Get Chunks to decompress
        chunks = get_chunks_for_slice(zarr_array, slices)

        # Get Compressor
        compressor = zarr_array.compressor

        # Make sure chunks are on correct device
        if self.device == "gpu":
            chunks = [cp.array(chunk) for chunk in chunks]

        # Sync before starting timer
        cp.cuda.runtime.deviceSynchronize()

        # Start timer
        tic = time.time()

        # Decompress chunks
        for _ in range(nr_repeats):
            if self.batch_codec:
                compressor.decode_batch(chunks)
            else:
                [compressor.decode(c) for c in chunks]

        # Sync after stopping timer
        cp.cuda.runtime.deviceSynchronize()

        # Stop timer
        toc = time.time()

        # Slice size in bytes
        slice_size = (
            np.prod([s.stop - s.start for s in slices]) * zarr_array.dtype.itemsize
        )

        # Return GB/s
        return (slice_size * nr_repeats) / (toc - tic) / 1e9

    def get_throughput(self, slices, comm):
        # Get rank and size
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Get array
        array = self.open_array()

        # Get slices for each rank
        dim_0_size = slices[0].stop - slices[0].start
        total_slices_in_array = int(array.shape[0] // dim_0_size)
        slices_per_rank = int(total_slices_in_array // size)

        # Sync before starting timer
        cp.cuda.runtime.deviceSynchronize()
        comm.Barrier()

        # Start timer
        tic = time.time()

        # Load slices
        for i in range(slices_per_rank):
            slice_dat = [
                slice(
                    dim_0_size * i + rank * dim_0_size * slices_per_rank,
                    dim_0_size * (i + 1) + rank * dim_0_size * slices_per_rank,
                )
            ] + slices[1:]
            a = array[tuple(slice_dat)]

        # Get total time
        cp.cuda.runtime.deviceSynchronize()
        toc = time.time()
        total_time = toc - tic

        # Sync all threads stopping timer
        comm.Barrier()

        # Get average time
        total_thread_time = comm.reduce(total_time, op=MPI.SUM)

        if rank == 0:
            avg_time = total_thread_time / size

            # Compute total pulled data
            slice_size = (
                np.prod([s.stop - s.start for s in slices])
                * array.dtype.itemsize
                * slices_per_rank
                * size
            )

            # Return GB/s
            return (slice_size) / avg_time / 1e9
        else:
            return None
