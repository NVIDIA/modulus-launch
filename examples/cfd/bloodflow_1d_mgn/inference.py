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

import torch, dgl
from dgl.dataloading import GraphDataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import tri as mtri
import os
from matplotlib.patches import Rectangle

from generate_dataset import generate_normalized_graphs
from generate_dataset import train_test_split
from generate_dataset import Bloodflow1DDataset
from modulus.models.meshgraphnet import MeshGraphNet
from modulus.launch.logging import PythonLogger
from modulus.launch.utils import load_checkpoint
import hydra
from omegaconf import DictConfig, OmegaConf
import json

class MGNRollout:
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

        test_graphs = [graphs[gname] for gname in testset]
        self.dataset = Bloodflow1DDataset(test_graphs, params, testset)

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=1,
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

        # enable eval mode
        self.model.eval()

        # load checkpoint
        _ = load_checkpoint(
            os.path.join(cfg.checkpoints.ckpt_path, 
                         cfg.checkpoints.ckpt_name),
            models=self.model,
            device=self.device,
        )

        self.var_identifier = {"p": 0, "q": 1}

    def predict(self, idx):
        self.pred, self.exact = [], []
        
        params = json.load(open('checkpoints/parameters.json'))

        # for i, graph in enumerate(self.dataloader):
        #     graph = graph.to(self.device)
        #     # denormalize data
        #     graph.ndata["x"][:, 0:2] = self.dataset.denormalize(
        #         graph.ndata["x"][:, 0:2], stats["velocity_mean"], 
        #         stats["velocity_std"]
        #     )
        #     graph.ndata["y"][:, 0:2] = self.dataset.denormalize(
        #         graph.ndata["y"][:, 0:2],
        #         stats["velocity_diff_mean"],
        #         stats["velocity_diff_std"],
        #     )

        #     # inference step
        #     invar = graph.ndata["x"].clone()

        #     if i != 0:
        #         invar[:, 0:2] = self.pred[i - 1][:, 0:2].clone()

        #     pred_i = self.model(invar, graph.edata["x"], graph).detach()  # predict

        #     # denormalize prediction
        #     pred_i[:, 0:2] = self.dataset.denormalize(
        #         pred_i[:, 0:2], stats["velocity_diff_mean"], stats["velocity_diff_std"]
        #     )
        #     pred_i[:, 2] = self.dataset.denormalize(
        #         pred_i[:, 2], stats["pressure_mean"], stats["pressure_std"]
        #     )
        #     invar[:, 0:2] = self.dataset.denormalize(
        #         invar[:, 0:2], stats["velocity_mean"], stats["velocity_std"]
        #     )

        #     # do not update the "wall_boundary" & "outflow" nodes
        #     mask = torch.cat((mask, mask), dim=-1).to(self.device)
        #     pred_i[:, 0:2] = torch.where(
        #         mask, pred_i[:, 0:2], torch.zeros_like(pred_i[:, 0:2])
        #     )

        #     # integration
        #     self.pred.append(
        #         torch.cat(
        #             ((pred_i[:, 0:2] + invar[:, 0:2]), pred_i[:, [2]]), dim=-1
        #         ).cpu()
        #     )
        #     self.exact.append(
        #         torch.cat(
        #             (
        #                 (graph.ndata["y"][:, 0:2] + graph.ndata["x"][:, 0:2]),
        #                 graph.ndata["y"][:, [2]],
        #             ),
        #             dim=-1,
        #         ).cpu()
        #     )

        #     self.faces.append(torch.squeeze(cells).numpy())
        #     self.graphs.append(graph.cpu())

        # # keep the QoI only
        # self.pred = [var[:, idx] for var in self.pred]
        # self.exact = [var[:, idx] for var in self.exact]

@hydra.main(version_base = None, config_path = ".", config_name = "config") 
def do_rollout(cfg: DictConfig):
    logger = PythonLogger("main")
    logger.file_logging()
    logger.info("Rollout started...")
    rollout = MGNRollout(logger, cfg)
    # for i in idx:
    #     rollout.predict(i)
    #     rollout.init_animation()
    #     ani = animation.FuncAnimation(
    #         rollout.fig,
    #         rollout.animate,
    #         frames=len(rollout.graphs) // C.frame_skip,
    #         interval=C.frame_interval,
    #     )
    #     ani.save("animations/animation_" + C.viz_vars[i] + ".gif")
    #     logger.info(f"Completed rollout for {C.viz_vars[i]}")


if __name__ == "__main__":
    do_rollout()