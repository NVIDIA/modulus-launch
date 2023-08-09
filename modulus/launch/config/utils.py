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
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from typing import Union


def to_absolute_path(*args: Union[str, Path]):
    """Converts file path to absolute path based on run file location
    Modified from: https://github.com/facebookresearch/hydra/blob/main/hydra/utils.py
    """
    out = ()
    for path in args:
        p = Path(path)
        if not HydraConfig.initialized():
            base = Path(os.getcwd())
        else:
            ret = HydraConfig.get().runtime.cwd
            base = Path(ret)
        if p.is_absolute():
            ret = p
        else:
            ret = base / p

        if isinstance(path, str):
            out = out + (str(ret),)
        else:
            out = out + (ret,)

    if len(args) == 1:
        out = out[0]
    return out
