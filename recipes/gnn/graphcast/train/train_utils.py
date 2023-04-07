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


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, model, optimizer, scheduler=None, scaler=None, iter=0):
    # create state dict
    state_dict = {
        "iter": iter,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()
    if scaler:
        state_dict["scaler_state_dict"] = scaler.state_dict()

    # save checkpoint
    torch.save(state_dict, path)
    print(f"saved model in {path}")


def load_checkpoint(
    path, model, optimizer, scheduler=None, scaler=None, map_location=None
):
    # load checkpoint
    try:
        checkpoint = torch.load(path, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        iter_init = checkpoint["iter"]
        print(f"Successfully loaded checkpoint in {path}")
    except:
        iter_init = 1
        print(f"Failed loading checkpoint in {path}")
    return model, optimizer, scheduler, scaler, iter_init


def make_dir(dir):
    if os.path.exists(dir):
        pass
    else:
        os.makedirs(dir)


def rprint(dist, msg):
    """Prints a message on rank 0"""
    if dist.rank == 0:
        print(msg)
