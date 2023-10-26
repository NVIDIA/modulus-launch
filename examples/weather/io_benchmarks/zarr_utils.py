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

import zarr
import cupy as cp
import numpy as np

def get_chunks_for_slice(zarr_array, slices):
    """
    Given a Zarr array and a tuple of slices, return the keys for all chunks that
    intersect with the slices.
    """

    # Calculate chunk indices for each dimension using numpy
    starts = np.array([s.start if s.start is not None else 0 for s in slices])
    stops = np.array(
        [
            s.stop if s.stop is not None else dim
            for s, dim in zip(slices, zarr_array.shape)
        ]
    )
    chunk_lens = np.array(zarr_array.chunks)

    start_idxs = (starts // chunk_lens).tolist()
    stop_idxs = (-(-stops // chunk_lens)).tolist()

    # Generate chunk ranges
    chunk_ranges = [
        np.arange(start, stop) for start, stop in zip(start_idxs, stop_idxs)
    ]

    # Use meshgrid to generate combinations of chunk indices
    grids = np.meshgrid(*chunk_ranges, indexing="ij")

    # Reshape and stack to get all combinations
    combined_indices = np.stack(grids, axis=-1).reshape(-1, len(chunk_ranges))

    # Generate keys
    keys = [
        str(tuple(row)).replace(", ", ".").replace("(", "").replace(")", "")
        for row in combined_indices
    ]

    # Get chunks from store
    chunks = [value for value in zarr_array.chunk_store.getitems(keys).values()]

    return chunks
