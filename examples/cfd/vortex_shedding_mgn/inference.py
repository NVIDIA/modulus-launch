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

import math
import os

import matplotlib.pyplot as plt
import torch
from dgl.dataloading import GraphDataLoader
from matplotlib import animation
from matplotlib import tri as mtri
from matplotlib.patches import Rectangle
from modulus.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset
from modulus.models.meshgraphnet import MeshGraphNet

from constants import Constants
from modulus.launch.logging import PythonLogger
from modulus.launch.utils import load_checkpoint

# Instantiate constants
C = Constants()


class MGNRollout:
    def __init__(self, logger):
        # set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {self.device} device")

        # instantiate dataset
        self.dataset = VortexSheddingDataset(
            name="vortex_shedding_test",
            data_dir=C.data_dir,
            split="test",
            num_samples=C.num_test_samples,
            num_steps=C.num_test_time_steps,
        )

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=1,  # TODO add support for batch_size > 1
            shuffle=False,
            drop_last=False,
        )

        # instantiate the model
        self.model = MeshGraphNet(
            C.num_input_features, C.num_edge_features, C.num_output_features
        )
        if C.jit:
            self.model = torch.jit.script(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)

        # instantiate loss
        self.criterion = torch.nn.MSELoss()

        # enable train mode
        self.model.eval()

        # load checkpoint
        _ = load_checkpoint(
            os.path.join(C.ckpt_path, C.ckpt_name),
            models=self.model,
            device=self.device,
        )

        self.var_identifier = {"u": 0, "v": 1, "p": 2}

    def predict(self):
        self.pred, self.exact, self.faces, self.graphs, self.loss = [], [], [], [], []
        stats = {
            key: value.to(self.device) for key, value in self.dataset.node_stats.items()
        }
        for i, (graph, cells, mask) in enumerate(self.dataloader):
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
            graph.ndata["y"][:, [2]] = self.dataset.denormalize(
                graph.ndata["y"][:, [2]],
                stats["pressure_mean"],
                stats["pressure_std"],
            )

            # inference step
            invar = graph.ndata["x"].clone()

            if i % (C.num_test_time_steps - 1) != 0:
                invar[:, 0:2] = self.pred[i - 1][:, 0:2].clone()
                i += 1
            invar[:, 0:2] = self.dataset.normalize_node(
                invar[:, 0:2], stats["velocity_mean"], stats["velocity_std"]
            )

            # Get the predition
            pred_i = self.model(invar, graph.edata["x"], graph).detach()  # predict

            # denormalize prediction
            pred_i[:, 0:2] = self.dataset.denormalize(
                pred_i[:, 0:2], stats["velocity_diff_mean"], stats["velocity_diff_std"]
            )
            pred_i[:, 2] = self.dataset.denormalize(
                pred_i[:, 2], stats["pressure_mean"], stats["pressure_std"]
            )

            loss = self.criterion(pred_i, graph.ndata["y"])
            self.loss.append(loss.cpu().detach())

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

    def init_animation(self, idx):
        self.animation_variable = C.viz_vars[idx]
        self.pred_i = [var[:, idx] for var in self.pred]
        self.exact_i = [var[:, idx] for var in self.exact]

        # fig configs
        plt.rcParams["image.cmap"] = "inferno"
        self.fig, self.ax = plt.subplots(3, 1, figsize=(16, (9 / 2) * 3))

        # Set background color to black
        self.fig.set_facecolor("black")
        self.ax[0].set_facecolor("black")
        self.ax[1].set_facecolor("black")
        self.ax[2].set_facecolor("black")
        self.first_call = True
        self.text = None

        # make animations dir
        if not os.path.exists("./animations"):
            os.makedirs("./animations")

    def animate(self, num):
        # Setup the colour bar ranges
        if self.animation_variable == "u":
            min_var = -1.0
            max_var = 4.0
            min_delta_var = -0.25
            max_delta_var = 0.25
        elif self.animation_variable == "v":
            min_var = -2.0
            max_var = 2.0
            min_delta_var = -0.25
            max_delta_var = 0.25
        elif self.animation_variable == "p":
            min_var = -5.0
            max_var = 5.0
            min_delta_var = -0.25
            max_delta_var = 0.25

        num *= C.frame_skip
        graph = self.graphs[num]
        y_star = self.pred_i[num].numpy()
        y_exact = self.exact_i[num].numpy()
        y_error = y_star - y_exact
        triang = mtri.Triangulation(
            graph.ndata["mesh_pos"][:, 0].numpy(),
            graph.ndata["mesh_pos"][:, 1].numpy(),
            self.faces[num],
        )

        # Prediction plotting
        self.ax[0].cla()
        self.ax[0].set_aspect("equal")
        self.ax[0].set_axis_off()
        navy_box = Rectangle((0, 0), 1.4, 0.4, facecolor="navy")
        self.ax[0].add_patch(navy_box)  # Add a navy box to the first subplot
        tripcolor_plot = self.ax[0].tripcolor(triang, y_star, vmin=min_var, vmax=max_var)
        self.ax[0].triplot(triang, "ko-", ms=0.5, lw=0.3)
        self.ax[0].set_title("Modulus MeshGraphNet Prediction", color="white")
        if num == 0 and self.first_call:
            cb_ax = self.fig.add_axes([0.9525, 0.69, 0.01, 0.26])
            self.setup_colourbars(tripcolor_plot, cb_ax)

        # Update the text for the example number and step number
        example_num = math.floor(num / C.num_test_time_steps)
        if self.text is None:
            self.text = plt.text(0.001, 0.9,
                                 f"Example {example_num + 1}: "
                                 f"{num - example_num * C.num_test_time_steps}/{C.num_test_time_steps}",
                                 color="white", fontsize=20, transform=self.ax[0].transAxes)
        else:
            self.text.set_text(
                f"Example {example_num + 1}: {num - example_num * C.num_test_time_steps}/{C.num_test_time_steps}")

        # Truth plotting
        self.ax[1].cla()
        self.ax[1].set_aspect("equal")
        self.ax[1].set_axis_off()
        navy_box = Rectangle((0, 0), 1.4, 0.4, facecolor="navy")
        self.ax[1].add_patch(navy_box)  # Add a navy box to the second subplot
        tripcolor_plot = self.ax[1].tripcolor(triang, y_exact, vmin=min_var, vmax=max_var)
        self.ax[1].triplot(triang, "ko-", ms=0.5, lw=0.3)
        self.ax[1].set_title("Ground Truth", color="white")
        if num == 0 and self.first_call:
            cb_ax = self.fig.add_axes([0.9525, 0.37, 0.01, 0.26])
            self.setup_colourbars(tripcolor_plot, cb_ax)

        # Error plotting
        self.ax[2].cla()
        self.ax[2].set_aspect("equal")
        self.ax[2].set_axis_off()
        navy_box = Rectangle((0, 0), 1.4, 0.4, facecolor="navy")
        self.ax[2].add_patch(navy_box)  # Add a navy box to the second subplot
        tripcolor_plot = self.ax[2].tripcolor(
            triang, y_error, vmin=min_delta_var, vmax=max_delta_var, cmap="coolwarm"
        )
        self.ax[2].triplot(triang, "ko-", ms=0.5, lw=0.3)
        self.ax[2].set_title(
            "Absolute Error (Prediction - Ground Truth)", color="white"
        )
        if num == 0 and self.first_call:
            cb_ax = self.fig.add_axes([0.9525, 0.055, 0.01, 0.26])
            self.setup_colourbars(tripcolor_plot, cb_ax)

        # Adjust subplots to minimize empty space
        self.ax[0].set_aspect("auto", adjustable="box")
        self.ax[0].autoscale(enable=True, tight=True)

        self.ax[1].set_aspect("auto", adjustable="box")
        self.ax[1].autoscale(enable=True, tight=True)

        self.ax[2].set_aspect("auto", adjustable="box")
        self.ax[2].autoscale(enable=True, tight=True)

        self.fig.subplots_adjust(
            left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.2
        )
        return self.fig

    def setup_colourbars(self, tripcolor_plot, cb_ax):
        cb = self.fig.colorbar(tripcolor_plot, orientation="vertical", cax=cb_ax)
        cb.set_label(self.animation_variable, color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        cb.outline.set_edgecolor("white")
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")


if __name__ == "__main__":
    logger = PythonLogger("main")  # General python logger
    logger.file_logging()
    logger.info("Rollout started...")
    rollout = MGNRollout(logger)
    idx = [rollout.var_identifier[k] for k in C.viz_vars]
    rollout.predict()
    for i in idx:
        rollout.init_animation(i)
        ani = animation.FuncAnimation(
            rollout.fig,
            rollout.animate,
            frames=len(rollout.graphs) // C.frame_skip,
            interval=C.frame_interval,
        )
        ani.save("animations/animation_" + C.viz_vars[i] + ".gif", dpi=50)
        logger.info(f"Created animation for {C.viz_vars[i]}")

    # Plot the losses
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(16, 4.5))
    ax.set_title("Rollout loss")
    for i in range(C.num_test_samples):
        start = i * (C.num_test_time_steps - 1)
        end = i * (C.num_test_time_steps - 1) + (C.num_test_time_steps - 1)
        ax.plot(rollout.loss[start:end])
    ax.set_xlim([0, C.num_test_time_steps])
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Step loss")
    plt.savefig("animations/rollout_loss.png")
