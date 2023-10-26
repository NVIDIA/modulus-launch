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

import os.path
from typing import Any, Literal, Mapping, Optional, Sequence, Union
import cupy as cp

import numcodecs
import numpy
import numpy as np
import zarr
import zarr.creation
import zarr.storage
from numcodecs.compat import ensure_contiguous_ndarray_like
from numcodecs.registry import register_codec
from packaging.version import parse

import kvikio
import kvikio.nvcomp
import kvikio.nvcomp_codec
import kvikio.zarr
from kvikio.nvcomp_codec import NvCompBatchCodec

class MemMapStore(zarr.storage.DirectoryStore):
    """Loads zarr data via memmap.

    This class works like `zarr.storage.DirectoryStore` but implements
    getitems() using memmap instead of reading the data into memory.
    Depending on the system this can give throughput improvements.
    This class is currently only used for testing and should not be
    used in production.

    Parameters
    ----------
    path : string
        Location of directory to use as the root of the storage hierarchy.
    """

    # The default output array type used by getitems().
    default_meta_array = cp.empty(())

    def __init__(
        self,
        path,
    ) -> None:
        super().__init__(path)

    def getitems(
        self,
        keys: Sequence[str],
        contexts: Mapping[str, Mapping] = {},
    ) -> Mapping[str, Any]:
        """Retrieve data from multiple keys using memmap.

        Parameters
        ----------
        keys : Iterable[str]
            The keys to retrieve
        contexts: Mapping[str, Context]
            A mapping of keys to their context. Each context is a mapping of store
            specific information. If the "meta_array" key exist, GDSStore use its
            values as the output array otherwise GDSStore.default_meta_array is used.

        Returns
        -------
        Mapping
            A collection mapping the input keys to their results.
        """
        ret = {}

        for key in keys:
            filepath = os.path.join(self.path, key)
            if not os.path.isfile(filepath):
                continue
            try:
                meta_array = contexts[key]["meta_array"]
            except KeyError:
                meta_array = self.default_meta_array

            nbytes = os.path.getsize(filepath)
            memmape = np.memmap(filepath, dtype="u1", mode="r", shape=(nbytes,))
            ret[key] = cp.asarray(memmape)

        return ret
