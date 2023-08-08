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

from modulus.models.meshgraphnet import MeshGraphNet
from modulus.launch.logging import PythonLogger
from modulus.launch.utils import load_checkpoint
from constants import Constants

import json

# Instantiate constants
C = Constants()


class MGNRollout:
    def __init__(self, logger):
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

        test_graphs = [graphs[gname] for gname in trainset]
        self.dataset = Bloodflow1DDataset(train_graphs, params, test_graphs)

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
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

        # enable train mode
        self.model.eval()

        # load checkpoint
        _ = load_checkpoint(
            os.path.join(C.ckpt_path, C.ckpt_name),
            models=self.model,
            device=self.device,
        )

        self.var_identifier = {"p": 0, "q": 1}

    def predict(self, idx):
        self.pred, self.exact = [], []
        
        params = json.load(open('checkpoints/parameters.json'))

        for i, graph in enumerate(self.dataloader):
            graph = graph.to(self.device)
            # denormalize data
            graph.ndata["x"][:, 0:2] = self.dataset.denormalize(
                graph.ndata["x"][:, 0:2], stats["velocity_mean"], stats["velocity_std"]
            )
            graph.ndata["y"][:, 0:2] = self.dataset.denormalize(
                graph.ndata["y"][:, 0:2],
                stats["velocity_diff_mean"],
                stats["velocity_diff_std"],
            )

            # inference step
            invar = graph.ndata["x"].clone()

            if i % (C.num_test_time_steps - 1) != 0:
                invar[:, 0:2] = self.pred[i - 1][:, 0:2].clone()
                i += 1
            invar[:, 0:2] = self.dataset.normalize_node(
                invar[:, 0:2], stats["velocity_mean"], stats["velocity_std"]
            )
            pred_i = self.model(invar, graph.edata["x"], graph).detach()  # predict

            # denormalize prediction
            pred_i[:, 0:2] = self.dataset.denormalize(
                pred_i[:, 0:2], stats["velocity_diff_mean"], stats["velocity_diff_std"]
            )
            pred_i[:, 2] = self.dataset.denormalize(
                pred_i[:, 2], stats["pressure_mean"], stats["pressure_std"]
            )
            invar[:, 0:2] = self.dataset.denormalize(
                invar[:, 0:2], stats["velocity_mean"], stats["velocity_std"]
            )

            # do not update the "wall_boundary" & "outflow" nodes
            mask = torch.cat((mask, mask), dim=-1).to(self.device)
            pred_i[:, 0:2] = torch.where(
                mask, pred_i[:, 0:2], torch.zeros_like(pred_i[:, 0:2])
            )

            # integration
            self.pred.append(
                torch.cat(
                    ((pred_i[:, 0:2] + invar[:, 0:2]), pred_i[:, [2]]), dim=-1
                ).cpu()
            )
            self.exact.append(
                torch.cat(
                    (
                        (graph.ndata["y"][:, 0:2] + graph.ndata["x"][:, 0:2]),
                        graph.ndata["y"][:, [2]],
                    ),
                    dim=-1,
                ).cpu()
            )

            self.faces.append(torch.squeeze(cells).numpy())
            self.graphs.append(graph.cpu())

        # keep the QoI only
        self.pred = [var[:, idx] for var in self.pred]
        self.exact = [var[:, idx] for var in self.exact]

    # def init_animation(self):
    #     # fig configs
    #     plt.rcParams["image.cmap"] = "inferno"
    #     self.fig, self.ax = plt.subplots(2, 1, figsize=(16, 9))

    #     # Set background color to black
    #     self.fig.set_facecolor("black")
    #     self.ax[0].set_facecolor("black")
    #     self.ax[1].set_facecolor("black")

    #     # make animations dir
    #     if not os.path.exists("./animations"):
    #         os.makedirs("./animations")

    # def animate(self, num):
    #     num *= C.frame_skip
    #     graph = self.graphs[num]
    #     y_star = self.pred[num].numpy()
    #     y_exact = self.exact[num].numpy()
    #     triang = mtri.Triangulation(
    #         graph.ndata["mesh_pos"][:, 0].numpy(),
    #         graph.ndata["mesh_pos"][:, 1].numpy(),
    #         self.faces[num],
    #     )
    #     self.ax[0].cla()
    #     self.ax[0].set_aspect("equal")
    #     self.ax[0].set_axis_off()
    #     navy_box = Rectangle((0, 0), 1.4, 0.4, facecolor="navy")
    #     self.ax[0].add_patch(navy_box)  # Add a navy box to the first subplot
    #     self.ax[0].tripcolor(triang, y_star, vmin=np.min(y_star), vmax=np.max(y_star))
    #     self.ax[0].triplot(triang, "ko-", ms=0.5, lw=0.3)
    #     self.ax[0].set_title("Modulus MeshGraphNet Prediction", color="white")
    #     self.ax[1].cla()
    #     self.ax[1].set_aspect("equal")
    #     self.ax[1].set_axis_off()
    #     navy_box = Rectangle((0, 0), 1.4, 0.4, facecolor="navy")
    #     self.ax[1].add_patch(navy_box)  # Add a navy box to the second subplot
    #     self.ax[1].tripcolor(
    #         triang, y_exact, vmin=np.min(y_exact), vmax=np.max(y_exact)
    #     )
    #     self.ax[1].triplot(triang, "ko-", ms=0.5, lw=0.3)
    #     self.ax[1].set_title("Ground Truth", color="white")

    #     # Adjust subplots to minimize empty space
    #     self.ax[0].set_aspect("auto", adjustable="box")
    #     self.ax[1].set_aspect("auto", adjustable="box")
    #     self.ax[0].autoscale(enable=True, tight=True)
    #     self.ax[1].autoscale(enable=True, tight=True)
    #     self.fig.subplots_adjust(
    #         left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.2
    #     )
    #     return self.fig


if __name__ == "__main__":
    logger = PythonLogger("main")
    logger.file_logging()
    logger.info("Rollout started...")
    rollout = MGNRollout(logger)
    idx = [rollout.var_identifier[k] for k in C.viz_vars]
    for i in idx:
        rollout.predict(i)
        rollout.init_animation()
        ani = animation.FuncAnimation(
            rollout.fig,
            rollout.animate,
            frames=len(rollout.graphs) // C.frame_skip,
            interval=C.frame_interval,
        )
        ani.save("animations/animation_" + C.viz_vars[i] + ".gif")
        logger.info(f"Completed rollout for {C.viz_vars[i]}")
