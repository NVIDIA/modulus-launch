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

import torch
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
import time, os
import wandb as wb

try:
    import apex
except:
    pass

from modulus.distributed.manager import DistributedManager
from modulus.models.meshgraphnet import MeshGraphNet
# from modulus.datapipes.gnn.mgn_dataset import MGNDataset
import generate_dataset as gd
from constants import Constants
from generate_dataset import generate_normalized_graphs
from generate_dataset import train_test_split
from generate_dataset import Bloodflow1DDataset

from modulus.launch.logging import (
    PythonLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint
import argparse

# Instantiate constants
C = Constants()

def parse_command_line_arguments():
    """
    Parse command line arguments.

    Returns:
        Data structure containing all the arguments
    """

    # parse arguments from command line
    parser = argparse.ArgumentParser(description='Graph Reduced Order Models')

    parser.add_argument('--bs', help='batch size', type=int, default=100)
    parser.add_argument('--epochs', help='total number of epochs', type=int,
                        default=100)
    parser.add_argument('--lr_decay', help='learning rate decay', type=float,
                        default=0.001)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--rate_noise', help='rate noise', type=float,
                        default=100)
    parser.add_argument('--rate_noise_features', help='rate noise features', 
                        type=float, default=1e-5)
    parser.add_argument('--weight_decay', help='l2 regularization', 
                        type=float, default=1e-5)
    parser.add_argument('--ls_gnn', help='latent size gnn', type=int,
                        default=16)
    parser.add_argument('--ls_mlp', help='latent size mlps', type=int,
                        default=64)
    parser.add_argument('--process_iterations', help='gnn layers', type=int,
                        default=3)
    parser.add_argument('--hl_mlp', help='hidden layers mlps', type=int,
                        default=2)
    parser.add_argument('--label_norm', help='0: min_max, 1: normal, 2: none',
                        type=int, default=1)
    parser.add_argument('--stride', help='stride for multistep training',
                        type=int, default=5)
    parser.add_argument('--bcs_gnn', help='path to graph for bcs',
                        type=str, default='models_bcs/31.10.2022_01.35.31')
    args = parser.parse_args()

    # we create a dictionary with all the parameters
    t_params = {'latent_size_gnn': args.ls_gnn,
                'latent_size_mlp': args.ls_mlp,
                'process_iterations': args.process_iterations,
                'number_hidden_layers_mlp': args.hl_mlp,
                'learning_rate': args.lr,
                'batch_size': args.bs,
                'lr_decay': args.lr_decay,
                'nepochs': args.epochs,
                'weight_decay': args.weight_decay,
                'rate_noise': args.rate_noise,
                'rate_noise_features': args.rate_noise_features,
                'stride': args.stride,
                'bcs_gnn': args.bcs_gnn}

    return t_params, args


class MGNTrainer:
    def __init__(self, wb, dist, rank_zero_logger):
        self.dist = dist

        t_params, args = parse_command_line_arguments()
        norm_type = {'features': 'normal', 'labels': 'normal'}
        graphs, params = generate_normalized_graphs('raw_dataset/graphs/',
                                                    norm_type)

        graph = graphs[list(graphs)[0]]

        infeat_nodes = graph.ndata['nfeatures'].shape[1] + 1
        infeat_edges = graph.edata['efeatures'].shape[1]
        nout = 2

        nodes_features = [
                'area', 
                'tangent', 
                'type',
                'T',
                'dip',
                'sysp',
                'resistance1',
                'capacitance',
                'resistance2',
                'loading']

        edges_features = [
            'rel_position', 
            'distance', 
            'type']

        t_params['infeat_nodes'] = infeat_nodes
        t_params['infeat_edges'] = infeat_edges
        t_params['out_size'] = nout
        params['node_features'] = nodes_features
        params['edges_features'] = edges_features

        params.update(t_params)

        trainset, testset = train_test_split(graphs, 0.9)

        train_graphs = [graphs[gname] for gname in trainset]
        traindataset = Bloodflow1DDataset(train_graphs, params, trainset)

        # test_graphs = [graphs[gname] for gname in testset]
        # traindataset = Bloodflow1DDataset(test_graphs, params, testset)

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            traindataset,
            batch_size=C.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=dist.world_size > 1,
        )

        # instantiate the model
        self.model = MeshGraphNet(
            17, params['infeat_edges'], 2
        )
        if C.jit:
            self.model = torch.jit.script(self.model).to(dist.device)
        else:
            self.model = self.model.to(dist.device)
        if C.watch_model and not C.jit and dist.rank == 0:
            wb.watch(self.model)

        # distributed data parallel for multi-node training
        if dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )

        # enable train mode
        self.model.train()

        # instantiate loss, optimizer, and scheduler
        self.criterion = torch.nn.MSELoss()
        try:
            self.optimizer = apex.optimizers.FusedAdam(self.model.parameters(), lr=C.lr)
            rank_zero_logger.info("Using FusedAdam optimizer")
        except:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=C.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: C.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        # load checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            os.path.join(C.ckpt_path, C.ckpt_name),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=dist.device,
        )

    def train(self, graph):
        graph = graph.to(self.dist.device)
        self.optimizer.zero_grad()
        loss = self.forward(graph)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def forward(self, graph):
        # forward pass
        with autocast(enabled=C.amp):
            pred = self.model(graph.ndata["nfeatures"], 
                              graph.edata["efeatures"], graph)
            loss = self.criterion(pred, graph.ndata["delta"])
            return loss

    def backward(self, loss):
        # backward pass
        if C.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # save constants to JSON file
    # if dist.rank == 0:
    #     os.makedirs(C.ckpt_path, exist_ok=True)
    #     with open(
    #         os.path.join(C.ckpt_path, C.ckpt_name.replace(".pt", ".json")), "w"
    #     ) as json_file:
    #         json_file.write(C.json(indent=4))

    # initialize loggers
    # initialize_wandb(
    #     project="Modulus-Launch",
    #     entity="Modulus",
    #     name="Vortex_Shedding-Training",
    #     group="Vortex_Shedding-DDP-Group",
    #     mode=C.wandb_mode,
    # )  # Wandb logger
    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    logger.file_logging()

    trainer = MGNTrainer(wb, dist, rank_zero_logger)
    start = time.time()
    rank_zero_logger.info("Training started...")
    for epoch in range(trainer.epoch_init, C.epochs):
        count = 0
        for graph in trainer.dataloader:
            print('{:}/{:}'.format(count, len(trainer.dataloader)))
            loss = trainer.train(graph)
            count = count+1
        print(f"epoch: {epoch}, loss: {loss:10.3e}, time per epoch: {(time.time()-start):10.3e}")
        rank_zero_logger.info(
            f"epoch: {epoch}, loss: {loss:10.3e}, time per epoch: {(time.time()-start):10.3e}"
        )
        wb.log({"loss": loss.detach().cpu()})

        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            save_checkpoint(
                os.path.join(C.ckpt_path, C.ckpt_name),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            logger.info(f"Saved model on rank {dist.rank}")
        start = time.time()
    rank_zero_logger.info("Training completed!")
