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
import argparse
import torch
import logging

from modulus.utils.sfno import logging_utils
from modulus.utils.sfno.YParams import YParams

# distributed computing stuff
from modulus.utils.sfno.distributed import comm

# import trainer
from trainer import Trainer

from parse_dataset_metada import parse_dataset_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fin_parallel_size",
        default=1,
        type=int,
        help="Input feature parallelization",
    )
    parser.add_argument(
        "--fout_parallel_size",
        default=1,
        type=int,
        help="Output feature parallelization",
    )
    parser.add_argument(
        "--h_parallel_size",
        default=1,
        type=int,
        help="Spatial parallelism dimension in h",
    )
    parser.add_argument(
        "--w_parallel_size",
        default=1,
        type=int,
        help="Spatial parallelism dimension in w",
    )
    parser.add_argument(
        "--parameters_reduction_buffer_count",
        default=1,
        type=int,
        help="How many buffers will be used (approximately) for weight gradient reductions.",
    )
    parser.add_argument("--run_num", default="00", type=str)
    parser.add_argument("--yaml_config", default="./config/sfnonet.yaml", type=str)
    parser.add_argument(
        "--batch_size",
        default=-1,
        type=int,
        help="Switch for overriding batch size in the configuration file.",
    )
    parser.add_argument("--config", default="default", type=str)
    parser.add_argument("--enable_synthetic_data", action="store_true")
    parser.add_argument(
        "--amp_mode",
        default="none",
        type=str,
        choices=["none", "fp16", "bf16"],
        help="Specify the mixed precision mode which should be used.",
    )
    parser.add_argument(
        "--jit_mode",
        default="none",
        type=str,
        choices=["none", "script", "inductor"],
        help="Specify if and how to use torch jit.",
    )
    parser.add_argument(
        "--cuda_graph_mode",
        default="none",
        type=str,
        choices=["none", "fwdbwd", "step"],
        help="Specify which parts to capture under cuda graph",
    )
    parser.add_argument("--enable_benchy", action="store_true")
    parser.add_argument("--disable_ddp", action="store_true")
    parser.add_argument("--enable_nhwc", action="store_true")
    parser.add_argument(
        "--checkpointing_level",
        default=0,
        type=int,
        help="How aggressively checkpointing is used",
    )
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--split_data_channels", action="store_true")
    parser.add_argument(
        "--print_timings_frequency",
        default=-1,
        type=int,
        help="Frequency at which to print timing information",
    )

    # checkpoint format
    parser.add_argument(
        "--save_checkpoint",
        default="distributed",
        type=str,
        choices=["distributed", "gathered"],
        help="Format in which to save checkpoints. Can be 'distributed' or 'gathered'",
    )
    parser.add_argument(
        "--load_checkpoint",
        default="distributed",
        type=str,
        choices=["distributed", "gathered"],
        help="Format in which to load checkpoints. Can be 'distributed' or 'gathered'",
    )

    # multistep stuff
    parser.add_argument(
        "--multistep_count",
        default=1,
        type=int,
        help="Number of autoregressive training steps. A value of 1 denotes conventional training",
    )

    # parse
    args = parser.parse_args()

    # parse parameters
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params["epsilon_factor"] = args.epsilon_factor

    # distributed
    params["fin_parallel_size"] = args.fin_parallel_size
    params["fout_parallel_size"] = args.fout_parallel_size
    params["h_parallel_size"] = args.h_parallel_size
    params["w_parallel_size"] = args.w_parallel_size

    params["model_parallel_sizes"] = [
        args.h_parallel_size,
        args.w_parallel_size,
        args.fin_parallel_size,
        args.fout_parallel_size,
    ]
    params["model_parallel_names"] = ["h", "w", "fin", "fout"]
    params["parameters_reduction_buffer_count"] = args.parameters_reduction_buffer_count

    # checkpoint format
    params["load_checkpoint"] = args.load_checkpoint
    params["save_checkpoint"] = args.save_checkpoint

    # make sure to reconfigure logger after the pytorch distributed init
    comm.init(params, verbose=False)

    world_rank = comm.get_world_rank()

    # update parameters
    params["world_size"] = comm.get_world_size()
    if args.batch_size > 0:
        params.batch_size = args.batch_size
    params["global_batch_size"] = params.batch_size
    assert (
        params["global_batch_size"] % comm.get_size("data") == 0
    ), f"Error, cannot evenly distribute {params['global_batch_size']} across {comm.get_size('data')} GPU."
    params["batch_size"] = int(params["global_batch_size"] // comm.get_size("data"))

    # optimizer params
    if "optimizer_max_grad_norm" not in params:
        params["optimizer_max_grad_norm"] = 1.0

    # set device
    torch.cuda.set_device(comm.get_local_rank())
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set up directory
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    if world_rank == 0:
        logging.info(f"writing output to {expDir}")
        if not os.path.isdir(expDir):
            os.makedirs(expDir, exist_ok=True)
            os.makedirs(os.path.join(expDir, "training_checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(expDir, "wandb"), exist_ok=True)

    params["experiment_dir"] = os.path.abspath(expDir)
    params["checkpoint_path"] = os.path.join(
        expDir, "training_checkpoints/ckpt_mp{mp_rank}.tar"
    )
    params["best_checkpoint_path"] = os.path.join(
        expDir, "training_checkpoints/best_ckpt_mp{mp_rank}.tar"
    )

    # check if all checkpoint files are there
    args.resuming = True
    for mp_rank in range(comm.get_size("model")):
        checkpoint_fname = params.checkpoint_path.format(mp_rank=mp_rank)
        if params["load_checkpoint"] == "distributed" or mp_rank < 1:
            args.resuming = args.resuming and os.path.isfile(checkpoint_fname)

    params["resuming"] = args.resuming
    params["amp_mode"] = args.amp_mode
    params["jit_mode"] = args.jit_mode
    params["cuda_graph_mode"] = args.cuda_graph_mode
    params["enable_benchy"] = args.enable_benchy
    params["disable_ddp"] = args.disable_ddp
    params["enable_nhwc"] = args.enable_nhwc
    params["checkpointing"] = args.checkpointing_level
    params["enable_synthetic_data"] = args.enable_synthetic_data
    params["split_data_channels"] = args.split_data_channels
    params["print_timings_frequency"] = args.print_timings_frequency
    params["multistep_count"] = args.multistep_count
    params["n_future"] = (
        args.multistep_count - 1
    )  # note that n_future counts only the additional samples

    # wandb configuration
    if params["wandb_name"] is None:
        params["wandb_name"] = args.config + "_" + str(args.run_num)
    if params["wandb_group"] is None:
        params["wandb_group"] = "era5_wind" + args.config
    if not hasattr(params, "wandb_dir") or params["swandb_dir"] is None:
        params["wandb_dir"] = expDir

    if world_rank == 0:
        logging_utils.log_to_file(
            logger_name=None, log_filename=os.path.join(expDir, "out.log")
        )
        logging_utils.log_versions()
        params.log()

    params["log_to_wandb"] = (world_rank == 0) and params["log_to_wandb"]
    params["log_to_screen"] = (world_rank == 0) and params["log_to_screen"]

    if "metadata_json_path" in params:
        params, _ = parse_dataset_metadata(params["metadata_json_path"], params=params)
    else:
        params["dhours"] = params.dhours if hasattr(params, "dhours") else 6
        params["channel_names"] = params.channel_names
        channels_idx = list(range(len(params.channel_names)))

        # set training channels
        params["in_channels"] = params.in_channels
        params["out_channels"] = params.out_channels
        params.N_in_channels = len(params.in_channels)
        params.N_out_channels = len(params.out_channels)

    if world_rank == 0:
        logging.info(f"Using channel names: {params.channel_names}")

    trainer = Trainer(params, world_rank)
    trainer.train()
