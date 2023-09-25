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

from typing import Tuple

from pydantic import BaseModel


class Constants(BaseModel):
    """vortex shedding constants"""
    # Model name
    model_name: str = "training"

    # data configs
    data_dir: str = "raw_dataset/cylinder_flow/cylinder_flow"

    # Dataset setup
    # Training
    # The dataset has 1000 examples
    num_training_samples: int = 400
    # The dataset has 600 time steps
    num_training_time_steps: int = 300
    # The paper suggests using noise std of 0.02
    training_noise_std: float = 0.02

    # Validation
    # The dataset has 100 examples
    num_valid_samples: int = 10
    # The dataset has 600 time steps
    num_valid_time_steps: int = 300

    # Test
    # The dataset has 100 examples
    num_test_samples: int = 10
    # The dataset has 600 time steps
    num_test_time_steps: int = 300

    # training configs
    epochs: int = 25
    training_batch_size: int = 11
    # For now the validation batch size must be 1
    valid_batch_size: int = 1

    # Training setup
    lr: float = 0.0001
    lr_decay_rate: float = 0.9999991
    ckpt_path: str = "checkpoints"
    ckpt_name: str = "model.pt"

    # Mesh Graph Net Setup
    num_input_features: int = 6
    num_edge_features: int = 3
    num_output_features: int = 3
    processor_size: int = 15
    num_layers_node_processor: int = 2
    num_layers_edge_processor: int = 2
    hidden_dim_processor: int = 128
    hidden_dim_node_encoder: int = 128
    num_layers_node_encoder: int = 2
    hidden_dim_edge_encoder: int = 128
    num_layers_edge_encoder: int = 2
    hidden_dim_node_decoder: int = 128
    num_layers_node_decoder: int = 2
    aggregation: str = "sum"
    do_concat_trick: bool = False
    num_processor_checkpoint_segments: int = 0
    # activation_fn: str = "relu"

    # performance configs
    amp: bool = False
    jit: bool = False

    # test & visualization configs
    viz_vars: Tuple[str, ...] = ("u", "v", "p")
    frame_skip: int = 10
    frame_interval: int = 1

    # wb configs
    wandb_mode: str = "disabled"
    watch_model: bool = False
