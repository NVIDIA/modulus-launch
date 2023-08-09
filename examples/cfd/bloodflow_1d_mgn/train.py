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
import numpy as np
import hydra

try:
    import apex
except:
    pass

from modulus.distributed.manager import DistributedManager
from modulus.models.meshgraphnet import MeshGraphNet
# from modulus.datapipes.gnn.mgn_dataset import MGNDataset
import generate_dataset as gd
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
import json
from omegaconf import DictConfig, OmegaConf

class MGNTrainer:
    def __init__(self, logger, cfg):
        # set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {self.device} device")

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

        params['infeat_nodes'] = infeat_nodes
        params['infeat_edges'] = infeat_edges
        params['out_size'] = nout
        params['node_features'] = nodes_features
        params['edges_features'] = edges_features
        params['rate_noise'] = 100
        params['rate_noise_features'] = 1e-5
        params['stride'] = 5

        trainset, testset = train_test_split(graphs, 0.9)

        train_graphs = [graphs[gname] for gname in trainset]
        traindataset = Bloodflow1DDataset(train_graphs, params, trainset)

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            traindataset,
            batch_size=cfg.training.batch_size,
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

        if cfg.performance.jit:
            self.model = torch.jit.script(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)

        # enable train mode
        self.model.train()

        # instantiate loss, optimizer, and scheduler
        self.criterion = torch.nn.MSELoss()
        try:
            self.optimizer = apex.optimizers.FusedAdam(self.model.parameters(), lr=cfg.scheduler.lr)
            rank_zero_logger.info("Using FusedAdam optimizer")
        except:
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                              lr=cfg.scheduler.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lr_lambda=lambda epoch: cfg.scheduler.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        # load checkpoint
        self.epoch_init = load_checkpoint(
            os.path.join(cfg.checkpoints.ckpt_path, cfg.checkpoints.ckpt_name),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
        )

        self.params = params
        self.cfg = cfg

    def backward(self, loss):
        # backward pass
        if self.cfg.checkpoints.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def train(self, graph):
        graph = graph.to(self.device)
        self.optimizer.zero_grad()
        loss = 0
        ns = graph.ndata['next_steps']

        # create mask to weight boundary nodes more in loss
        mask = torch.ones(ns[:,:,0].shape, device=self.device)
        imask = graph.ndata['inlet_mask'].bool()
        outmask = graph.ndata['outlet_mask'].bool()

        bcoeff = 100
        mask[imask,0] = mask[imask,0] * bcoeff 
        # flow rate is known 
        mask[outmask,0] = mask[outmask,0] * bcoeff
        mask[outmask,1] = mask[outmask,1] * bcoeff

        states = [graph.ndata["nfeatures"]]
        for istride in range(self.params['stride']):
            pred = self.model(graph, states[-1], 
                              graph.edata["efeatures"])

            # add prediction by MeshGraphNet to current state
            new_state = torch.clone(states[-1])
            new_state[:,0:2] += pred
            # impose exact flow rate at the inlet
            new_state[imask,1] = ns[imask,1,istride]
            states.append(new_state)

            coeff = 0.5
            if istride == 0:
                coeff = 1

            loss += coeff * self.criterion(states[-1][:,0:2], ns[:,:,istride])

        self.backward(loss)
        self.scheduler.step()

        def default(obj):
            if isinstance(obj, torch.Tensor):
                return default(obj.detach().numpy())
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.int64):
                return int(obj)
            print(obj)
            return TypeError('Token is not serializable')

        with open('checkpoints/parameters.json', 'w') as outfile:
            json.dump(self.params, outfile, default=default, indent=4)

        return loss

@hydra.main(version_base = None, config_path = ".", config_name = "config") 
def do_training(cfg: DictConfig):
    logger = PythonLogger("main")
    logger.file_logging()
    trainer = MGNTrainer(logger, cfg)
    start = time.time()
    logger.info("Training started...")
    for epoch in range(trainer.epoch_init, cfg.training.epochs):
        for graph in trainer.dataloader:
            loss = trainer.train(graph)

        logger.info(
            f"epoch: {epoch}, loss: {loss:10.3e}, time per epoch: {(time.time()-start):10.3e}"
        )

        # save checkpoint
        save_checkpoint(
                os.path.join(cfg.checkpoints.ckpt_path, 
                             cfg.checkpoints.ckpt_name),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
        )
        start = time.time()
    logger.info("Training completed!")

if __name__ == "__main__":
    do_training()