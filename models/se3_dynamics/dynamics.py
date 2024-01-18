import torch
import numpy as np
import dgl
from torch import nn
from torch import cdist

from ..se3_dynamics.models import OurSE3Transformer, SE3Transformer, TFN
from .utils.utils_profiling import * # load before other local modules


class OurDynamics(torch.nn.Module):
    def __init__(self, n_dimension, num_channel=16, n_layers=3, edge_dim = 1, out_dim = 3,  act_fn=nn.ReLU(inplace=True), model="se3_transformer", num_degrees=4, div=1):
        super().__init__()
        #self._transformation = transformation

        self._n_dimension = n_dimension   # the input feature dims:
        self.x_embedding = nn.Sequential(act_fn,
                                         nn.Linear(n_dimension, int(n_dimension / 2)),
                                         act_fn,
                                         nn.Linear(int(n_dimension / 2), int(n_dimension / 4)),
                                         act_fn,
                                         nn.Linear(int(n_dimension / 4), int(n_dimension / 8)),
                                         act_fn,
                                         nn.Linear(int(n_dimension / 8), 3),
                                         )


        if model == 'se3_transformer':
            self.se3 = SE3Transformer(num_layers = n_layers,  atom_feature_size = n_dimension, num_channels = num_channel,
                                 num_degrees = num_degrees, edge_dim = edge_dim, out_feat_dim = out_dim, div = div)

        elif model == 'tfn':
            self.se3 = SE3Transformer(num_layers=n_layers, atom_feature_size=n_dimension, num_channels=num_channel,
                                      num_degrees=num_degrees, edge_dim=edge_dim, out_feat_dim=out_dim, div=div)
        else:
            raise Exception("Wrong model")


        self.graph = None

    def forward(self, nodes, edges, edge_attr = None):

        # nodes.size() --> (num_nodes: n, dim: feature_size)
        # nodes = nodes.view(-1, self._n_particles, self._n_dimension)  # the new shape is: B*N*dim


        # 1. Transform nodes to G
        indices_src, indices_dst = edges
        x = self.x_embedding(nodes)
        distance = x[indices_src] - x[indices_dst]  # will has the same number edges as edge_attr all are equals len(indices_src)
        if self.graph is None:
            self.graph = dgl.graph((indices_src, indices_dst))
            self.graph.ndata['f'] = nodes[..., None] # [nodes, node_feature_dims, 1] and this is 0:feature that we must be used in se3
            self.graph.ndata['x'] = x  # [nodes, 3]    # x.unsqueeze(1) and this is 1: feature that if you want this feature to be used in se3
            self.graph.edata['w'] = edge_attr  # [num_edges, 1]
            self.graph.edata['d'] = distance   # [num_edges, 3]

        G = self.graph

        # 2. Transform G with se3t to G_out
        out = self.se3(G)  # out is the nodes coordinates

        # 3. Transform G_out to out
        distance = cdist(out, out, p=2)

        return out, out, distance

    '''
    @profile
    def f(self, nodes, edges, edge_attr):
        indices_src, indices_dst = edges
        x = self.x_embedding(nodes)
        distance = x[indices_src] - x[
            indices_dst]  # will has the same number edges as edge_attr all are equals len(indices_src)
        if self.graph is None:
            self.graph = dgl.DGLGraph((indices_src, indices_dst))
            self.graph.ndata['f'] = nodes[..., None]  # [nodes, node_feature_dims, 1] and this is 0:feature that we must be used in se3
            self.graph.ndata[
                'x'] = x  # [nodes, 3]    # x.unsqueeze(1) and this is 1: feature that if you want this feature to be used in se3
            self.graph.edata['w'] = edge_attr  # [num_edges, 1]
            self.graph.edata['d'] = distance  # [num_edges, 3]

        G = self.graph

        # 2. Transform G with se3t to G_out
        out = self.se3(G)  # out is the nodes coordinates

        return out
        '''



