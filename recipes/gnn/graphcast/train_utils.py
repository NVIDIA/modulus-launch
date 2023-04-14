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

import numpy as np
import torch
import os

class RankZeroLoggingWrapper:
    """Wrapper class to only log from rank 0 process in distributed training."""

    def __init__(self, obj, dist):
        self.obj = obj
        self.dist = dist

    def __getattr__(self, name):
        attr = getattr(self.obj, name)
        if callable(attr):

            def wrapper(*args, **kwargs):
                if self.dist.rank == 0:
                    return attr(*args, **kwargs)
                else:
                    return None

            return wrapper
        else:
            return attr


def count_trainable_params(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): Model to count parameters of.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
