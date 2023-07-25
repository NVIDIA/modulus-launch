# Copyright 2023 Stanford University

# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.

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

class MGNTrainer:
    def __init__(self, wb):
        self.device="cuda:0"

        t_params = {}
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

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            traindataset,
            batch_size=C.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=False,
        )
        # instantiate the model
        self.model = MeshGraphNet(
            17, 
            params['infeat_edges'], 
            2,
            processor_size=5,
            hidden_dim_node_encoder=64,
            hidden_dim_edge_encoder=64,
            hidden_dim_node_decoder=64   
        )

        if C.jit:
            self.model = torch.jit.script(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)
        if C.watch_model and not C.jit:
            wb.watch(self.model)

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
        self.epoch_init = load_checkpoint(
            os.path.join(C.ckpt_path, C.ckpt_name),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
        )

    def train(self, graph):
        graph = graph.to(self.device)
        self.optimizer.zero_grad()
        loss = self.forward(graph)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def forward(self, graph):
        # forward pass
        with autocast(enabled=C.amp):
            pred = self.model(graph, node_features = graph.ndata["nfeatures"], 
                              edge_features = graph.edata["efeatures"])
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
    trainer = MGNTrainer(wb)
    start = time.time()
    print("Training started...")
    for epoch in range(trainer.epoch_init, C.epochs):
        count = 0
        for graph in trainer.dataloader:
            loss = trainer.train(graph)
            count = count+1
        print(
            f"epoch: {epoch}, loss: {loss:10.3e}, time per epoch: {(time.time()-start):10.3e}"
        )

        # save checkpoint
        save_checkpoint(
                os.path.join(C.ckpt_path, C.ckpt_name),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
        )
        start = time.time()
    print("Training completed!")
