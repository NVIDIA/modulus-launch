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

import torch
import numpy as np
import pyvista as pv
import wandb as wb
from dgl.dataloading import GraphDataLoader
from dgl import DGLGraph

from modulus.models.meshgraphnet import MeshGraphNet
from modulus.launch.utils import load_checkpoint
from modulus.launch.logging import PythonLogger
from ahmed_body_dataset import AhmedBodyDataset
from constants import Constants

C = Constants()


def dgl_to_pyvista(graph: DGLGraph):
    """
    Converts a DGL graph to a PyVista graph.

    Parameters:
    -----------
    graph: DGLGraph
        The input DGL graph.

    Returns:
    --------
    pv_graph:
        The output PyVista graph.
    """

    # Convert the DGL graph to a NetworkX graph
    nx_graph = graph.to_networkx(
        node_attrs=["pos", "p_pred", "p", "s_pred", "wallShearStress"]
    ).to_undirected()

    # Initialize empty lists for storing data
    points = []
    lines = []
    p_pred = []
    s_pred = []
    p = []
    wallShearStress = []

    # Iterate over the nodes in the NetworkX graph
    for node, attributes in nx_graph.nodes(data=True):
        # Append the node and attribute data to the respective lists
        points.append(attributes["pos"].numpy())
        p_pred.append(attributes["p_pred"].numpy())
        s_pred.append(attributes["s_pred"].numpy())
        p.append(attributes["p"].numpy())
        wallShearStress.append(attributes["wallShearStress"].numpy())

    # Add edges to the lines list
    for edge in nx_graph.edges():
        lines.extend([2, edge[0], edge[1]])

    # Initialize a PyVista graph
    pv_graph = pv.PolyData()

    # Assign the points, lines, and attributes to the PyVista graph
    pv_graph.points = np.array(points)
    pv_graph.lines = np.array(lines)
    pv_graph.point_data["p_pred"] = np.array(p_pred)
    pv_graph.point_data["s_pred"] = np.array(s_pred)
    pv_graph.point_data["p"] = np.array(p)
    pv_graph.point_data["wallShearStress"] = np.array(wallShearStress)

    return pv_graph


class MGNRollout:
    def __init__(self, wb, logger):
        # set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {self.device} device")

        # instantiate dataset
        self.dataset = AhmedBodyDataset(
            name="ahmed_body_test",
            data_dir=C.data_dir,
            split="test",
            num_samples=C.num_test_samples,
            compute_drag=True,
        )

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=C.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # instantiate the model
        self.model = MeshGraphNet(
            C.input_dim_nodes,
            C.input_dim_edges,
            C.output_dim,
            aggregation=C.aggregation,
            hidden_dim_node_encoder=C.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=C.hidden_dim_edge_encoder,
            hidden_dim_node_decoder=C.hidden_dim_node_decoder,
        )
        self.model = self.model.to(self.device)

        # enable train mode
        self.model.eval()

        # load checkpoint
        _ = load_checkpoint(
            os.path.join(C.ckpt_path, C.ckpt_name),
            models=self.model,
            device=self.device,
        )

    def compute_drag_coefficient(self, mesh, velocity, frontal_area, p, s):
        """
        Compute drag coefficient for a given mesh.

        Parameters:
        -----------
        mesh: pyvista.PolyData:
            The input mesh
        velocity: float:
            Inlet velocity of the flow
        frontal_area: float:
            Frontal area of the body
        p: np.ndarray
            Pressure distribution on the mesh
        s: np.ndarray:
            Wall shear stress distribution on the mesh

        Returns:
        --------
        c_drag: float:
            Computed drag coefficient
        c_drag_pred: float
            Computed predicted drag coefficient
        """

        # Compute cell sizes and normals.
        mesh = mesh.compute_cell_sizes()
        mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)

        # Convert cell data to point data.
        mesh = mesh.cell_data_to_point_data()

        # Scale frontal area.
        frontal_area *= 10 ** (-6)

        # Compute coefficients.
        c_p = (2.0 / ((velocity**2) * frontal_area)) * np.sum(
            mesh["Normals"][:, 0] * mesh["Area"] * mesh["p"]
        )
        c_f = -(2.0 / ((velocity**2) * frontal_area)) * np.sum(
            mesh["wallShearStress"][:, 0] * mesh["Area"]
        )
        c_p_pred = (2.0 / ((velocity**2) * frontal_area)) * np.sum(
            mesh["Normals"][:, 0] * mesh["Area"] * p.reshape(-1)
        )
        c_f_pred = -(2.0 / ((velocity**2) * frontal_area)) * np.sum(
            s[:, 0] * mesh["Area"]
        )

        # Compute total drag coefficients.
        c_drag = c_p + c_f
        c_drag_pred = c_p_pred + c_f_pred

        return c_drag, c_drag_pred

    def denormalize(self, graph, stats):
        """
        Denormalize the graph data.

        Parameters:
        -----------
        graph: dgl.DGLGraph:
            Input graph
        stats: dict
            Statistics for normalization.

        Returns:
        --------
        graph: dgl.DGLGraph
            Graph with denormalized features
        """

        graph.ndata["p_pred"] = (graph.ndata["p_pred"] + 1) * (
            stats["p_max"] - stats["p_min"]
        ) / 2 + stats["p_min"]
        graph.ndata["s_pred"] = (graph.ndata["s_pred"] + 1) * (
            stats["wallShearStress_max"] - stats["wallShearStress_min"]
        ) / 2 + stats["wallShearStress_min"]
        graph.ndata["p"] = (graph.ndata["p"] + 1) * (
            stats["p_max"] - stats["p_min"]
        ) / 2 + stats["p_min"]
        graph.ndata["wallShearStress"] = (graph.ndata["wallShearStress"] + 1) * (
            stats["wallShearStress_max"] - stats["wallShearStress_min"]
        ) / 2 + stats["wallShearStress_min"]
        return graph

    def calculate_error(self, pred, y):
        """
        Calculate relative L2 error norm

        Parameters:
        -----------
        pred: torch.Tensor
            Predicted tensor
        y: torch.Tensor
            Ground truth tensor

        Returns:
        --------
        error: float
            Calculated relative L2 error norm (percentage)
        """

        error = torch.mean(torch.norm(pred - y, p=2) / torch.norm(y, p=2)).cpu().numpy()
        return error * 100

    def predict(self, save_results=False):
        """
        Run the prediction process.

        Parameters:
        -----------
        save_results: bool
            Whether to save the results in form of a .vtp file, by default False


        Returns:
        --------
        None
        """

        self.pred, self.exact, self.faces, self.graphs = [], [], [], []
        stats = {
            key: value.to(self.device) for key, value in self.dataset.node_stats.items()
        }

        for i, (graph, sid, velocity, frontal_area) in enumerate(self.dataloader):
            graph = graph.to(self.device)
            sid = sid.item()
            logger.info(f"Processing sample ID {sid}")
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph).detach()
            graph.ndata["p_pred"] = pred[:, [0]]
            graph.ndata["s_pred"] = pred[:, 1:]

            # denormalize
            graph = self.denormalize(graph, stats)
            pred = torch.concat((graph.ndata["p_pred"], graph.ndata["s_pred"]), dim=-1)
            y = torch.concat((graph.ndata["p"], graph.ndata["wallShearStress"]), dim=-1)

            error = self.calculate_error(pred, y)
            logger.info(f"Error percentage: {error}")

            # compute drag coefficient
            mesh = pv.read(
                os.path.join(C.data_dir, self.dataset.split, f"case{sid}.vtp")
            )
            p = graph.ndata["p_pred"].cpu().numpy()
            s = graph.ndata["s_pred"].cpu().numpy()
            c_d, c_d_pred = self.compute_drag_coefficient(
                mesh, velocity, frontal_area, p, s
            )

            logger.info(f"Ground truth Cd: {c_d}")
            logger.info(f"Predicted Cd: {c_d_pred}")

            if save_results:
                # Convert DGL graph to PyVista graph and save it
                os.makedirs(C.results_dir, exist_ok=True)
                pv_graph = dgl_to_pyvista(graph.cpu())
                pv_graph.save(os.path.join(C.results_dir, f"graph_{sid}.vtp"))


if __name__ == "__main__":
    logger = PythonLogger("main")  # General python logger
    logger.file_logging()

    logger.info("Rollout started...")
    rollout = MGNRollout(wb, logger)
    rollout.predict(save_results=True)
