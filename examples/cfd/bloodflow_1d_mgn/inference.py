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

import torch, dgl
from dgl.dataloading import GraphDataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from torch.cuda.amp import autocast, GradScaler
from generate_dataset import generate_normalized_graphs
from generate_dataset import train_test_split
from generate_dataset import Bloodflow1DDataset
from modulus.models.meshgraphnet import MeshGraphNet
from modulus.launch.logging import PythonLogger
from modulus.launch.utils import load_checkpoint
import hydra
from omegaconf import DictConfig, OmegaConf
import json
import time

def denormalize(tensor, mean, stdv):
        return tensor * stdv + mean 

class MGNRollout:
    def __init__(self, logger, cfg):
        # set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger
        logger.info(f"Using {self.device} device")

        params = json.load(open('checkpoints/parameters.json')) 

        norm_type = {'features': 'normal', 'labels': 'normal'}
        graphs, params = generate_normalized_graphs('raw_dataset/graphs/',
                                                    norm_type,
                                                    cfg.training.geometries,
                                                    params['statistics'])
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

        self.graphs = graphs

        # instantiate the model
        self.model = MeshGraphNet(
            params['infeat_nodes'], 
            params['infeat_edges'], 
            2,
            processor_size=5,
            hidden_dim_node_encoder=64,
            hidden_dim_edge_encoder=64,
            hidden_dim_processor=64,
            hidden_dim_node_decoder=64   
        )

        if cfg.performance.jit:
            self.model = torch.jit.script(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)

        self.scaler = GradScaler()
        # enable eval mode
        self.model.eval()

        # load checkpoint
        _ = load_checkpoint(
            os.path.join(cfg.checkpoints.ckpt_path, 
                         cfg.checkpoints.ckpt_name),
            models=self.model,
            device=self.device,
            scaler=self.scaler
        )

        self.params = params
        self.var_identifier = {"p": 0, "q": 1}

    def compute_average_branches(self, graph, flowrate):
        """
        Average flowrate over branch nodes

        Arguments:
            graph: DGL graph
            flowrate: 1D tensor containing nodal flow rate values

        """
        branch_id = graph.ndata['branch_id'].cpu().detach().numpy()
        bmax = np.max(branch_id)
        for i in range(bmax + 1):
            idxs = np.where(branch_id == i)[0]
            rflowrate = torch.mean(flowrate[idxs])
            flowrate[idxs] = rflowrate


    def predict(self, graph_name):
        """
        Perform rollout phase for a single graph in the dataset
    
        Arguments:
            graph_name: the graph name.
    
        """
        self.pred, self.exact = [], []
        
        graph = self.graphs[graph_name]
        graph = graph.to(self.device)
        self.graph = graph
        ntimes = graph.ndata['pressure'].shape[-1]

        inmask = graph.ndata['inlet_mask'].bool()
        invar = graph.ndata['nfeatures'][:,:,0].clone().squeeze()
        efeatures = graph.edata['efeatures'].squeeze()
        self.pred.append(invar[:,0:2].clone())
        self.exact.append(graph.ndata['nfeatures'][:,0:2,0])
        nnodes = inmask.shape[0]
        nf = torch.zeros((nnodes,1), device=self.device)
        start = time.time()
        for i in range(ntimes-1): 
            # set loading variable (check original paper for reference)
            invar[:,-1] = graph.ndata['nfeatures'][:,-1,i]
            # we set the next flow rate at the inlet (boundary condition)
            nf[inmask,0] = graph.ndata['nfeatures'][inmask,1,i+1]
            nfeatures = torch.cat((invar, nf), 1)
            pred = self.model(nfeatures, efeatures, graph).detach()
            invar[:,0:2] += pred            
            # we set the next flow rate at the inlet since that is known
            invar[inmask,1] = graph.ndata['nfeatures'][inmask,1,i+1]
            # flow rate must be constant in branches
            self.compute_average_branches(graph, invar[:,1])
            
            self.pred.append(invar[:,0:2].clone())
            self.exact.append(graph.ndata['nfeatures'][:,0:2,i+1])
         
        end = time.time()       
        self.logger.info(f"Rollout took {end - start} seconds!")   

    def denormalize(self):
        """
        Denormalize predicted and exact pressure and flow rate values. This 
        function must be called after 'predict'.
    
        Arguments:
            graph_name: the graph name.
    
        """
        nsol = len(self.pred)
        for isol in range(nsol):
            self.pred[isol][:,0] = denormalize(self.pred[isol][:,0],
                                  self.params['statistics']['pressure']['mean'],
                                  self.params['statistics']['pressure']['stdv'])
            self.pred[isol][:,1] = denormalize(self.pred[isol][:,1], 
                                  self.params['statistics']['flowrate']['mean'],
                                  self.params['statistics']['flowrate']['stdv'])
         
            self.exact[isol][:,0] = denormalize(self.exact[isol][:,0],
                                  self.params['statistics']['pressure']['mean'],
                                  self.params['statistics']['pressure']['stdv'])
            self.exact[isol][:,1] = denormalize(self.exact[isol][:,1], 
                                  self.params['statistics']['flowrate']['mean'],
                                  self.params['statistics']['flowrate']['stdv'])
    
    def compute_errors(self):
        """
        Compute errors in pressure and flow rate. This function must be called
        after 'predict' and 'denormalize'. The errors are computed as l2 errors
        at the branch nodes for all timesteps.

        """
        # compute errors
        branch_mask = self.graph.ndata['branch_mask'].bool()
        p_err = 0
        q_err = 0
        p_norm = 0
        q_norm = 0
        nsol = len(self.pred)
        for isol in range(nsol):
             p_pred = self.pred[isol][:,0]
             q_pred = self.pred[isol][:,1]
         
             p_exact = self.exact[isol][:,0]
             q_exact = self.exact[isol][:,1]
             p_err += torch.sum((p_pred[branch_mask] - p_exact[branch_mask])**2)
             q_err += torch.sum((q_pred[branch_mask] - q_exact[branch_mask])**2)

             p_norm += torch.sum(p_exact[branch_mask]**2)
             q_norm += torch.sum(q_exact[branch_mask]**2)

        # compute relative error
        p_err = torch.sqrt(p_err/p_norm)
        q_err = torch.sqrt(q_err/q_norm)

        self.logger.info(f"Relative error in pressure: {p_err * 100}%")
        self.logger.info(f"Relative error in flowrate: {q_err * 100}%")

    def plot(self, idx):
        """
        Creates plot of pressure and flow rate at the node specified with the 
        idx parameter.

        Arguments:
            idx: Index of the node to plot pressure and flow rate at.
        
        """
        load = self.graph.ndata['nfeatures'][0,-1,:]
        p_pred_values = []
        q_pred_values = []
        p_exact_values = []
        q_exact_values = []

        bm = self.graph.ndata['branch_mask'].bool()

        nsol = len(self.pred)
        for isol in range(nsol):
            if load[isol] == 0:
                p_pred_values.append(self.pred[isol][bm,0][idx].cpu())
                q_pred_values.append(self.pred[isol][bm,1][idx].cpu())
                p_exact_values.append(self.exact[isol][bm,0][idx].cpu())
                q_exact_values.append(self.exact[isol][bm,1][idx].cpu())

        plt.figure()
        ax = plt.axes()

        ax.plot(p_pred_values, label = 'pred')
        ax.plot(p_exact_values, label = 'exact')
        ax.legend()
        plt.savefig('pressure.png', bbox_inches='tight')

        plt.figure()
        ax = plt.axes()

        ax.plot(q_pred_values, label = 'pred')
        ax.plot(q_exact_values, label = 'exact')
        ax.legend()
        plt.savefig('flowrate.png', bbox_inches='tight')

@hydra.main(version_base = None, config_path = ".", config_name = "config") 
def do_rollout(cfg: DictConfig):
    """
    Perform rollout phase.

    Arguments:
        cfg: Dictionary containing problem parameters.
    
    """
    logger = PythonLogger("main")
    logger.file_logging()
    logger.info("Rollout started...")
    rollout = MGNRollout(logger, cfg)
    rollout.predict(cfg.testing.graph)
    rollout.denormalize()
    rollout.compute_errors()
    rollout.plot(idx = 5)

"""
The main function perform the rollout phase on the geometry specified in 
'config.yaml' (testing.graph) and computes the error.
"""
if __name__ == "__main__":
    do_rollout()

