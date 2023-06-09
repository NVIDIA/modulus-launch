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

DECORRELATION_TIME = 36  # 9 days

# distributed computing stuff
from modulus.utils.sfno.distributed import comm

# import trainer
from trainer import Trainer
from inferencer import Inferencer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matmul_parallel_size",
        default=1,
        type=int,
        help="Matmul parallelism dimension, only applicable to AFNO",
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
    parser.add_argument("--yaml_config", default="./config/afnonet.yaml", type=str)
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
        choices=["none", "script", "trace"],
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
    # for data prefetch buffers
    parser.add_argument(
        "--host_prefetch_buffers",
        action="store_true",
        default=False,
        help="Store file prefetch buffers on the host instead of the gpu, uses less GPU memory but can be slower",
    )
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--split_data_channels", action="store_true")
    parser.add_argument(
        "--print_timings_frequency",
        default=-1,
        type=int,
        help="Frequency at which to print timing information",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        default=False,
        help="Run inference instead of training",
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
    params["host_prefetch_buffers"] = args.host_prefetch_buffers

    # distributed
    params["matmul_parallel_size"] = args.matmul_parallel_size
    params["h_parallel_size"] = args.h_parallel_size
    params["w_parallel_size"] = args.w_parallel_size

    params["model_parallel_sizes"] = [
        args.h_parallel_size,
        args.w_parallel_size,
        args.matmul_parallel_size,
    ]
    params["model_parallel_names"] = ["h", "w", "matmul"]
    params["parameters_reduction_buffer_count"] = args.parameters_reduction_buffer_count

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

    # Do not comment this line out please:
    # check if all files are there
    args.resuming = True
    for mp_rank in range(comm.get_size("model")):
        checkpoint_fname = params.checkpoint_path.format(mp_rank=mp_rank)
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

    if world_rank == 0:
        logging_utils.log_to_file(
            logger_name=None, log_filename=os.path.join(expDir, "out.log")
        )
        logging_utils.log_versions()
        params.log()

    params["log_to_wandb"] = (world_rank == 0) and params["log_to_wandb"]
    params["log_to_screen"] = (world_rank == 0) and params["log_to_screen"]

    # instantiate trainer object
    if args.inference:
        inferencer = Inferencer(params, world_rank)
        inferencer.inference()
    else:
        trainer = Trainer(params, world_rank)
        trainer.train()
