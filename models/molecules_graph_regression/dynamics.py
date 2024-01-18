import torch
import numpy as np
import dgl
from torch import nn
from torch import cdist
from models.molecules_graph_regression.load_net import gnn_model
from ProcessData.molecules import self_loop as sl
from ProcessData.molecules import make_full_graph as mfg
from ProcessData.molecules import laplacian_positional_encoding as lpe
from ProcessData.molecules import wl_positional_encoding as wpe


class OurDynamics(torch.nn.Module):
    def __init__(self, mod_name = 'GraphTransformer', net_params = None):
        super().__init__()
        #self._transformation = transformation

        ''''
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
        '''
        self.lap = net_params['lap_pos_enc']
        self.wl_pos = net_params['wl_pos_enc']
        self.full = net_params['full_graph']
        self.pos_enc = net_params['pos_enc_dim']

        self.gnn = gnn_model(MODEL_NAME = mod_name, net_params = net_params)


        self.graph = None

    def forward(self, nodes, edges, edge_attr = None):

        # nodes.size() --> (num_nodes: n, dim: feature_size)
        # nodes = nodes.view(-1, self._n_particles, self._n_dimension)  # the new shape is: B*N*dim


        # 1. Transform nodes to G
        node_num = len(nodes)
        nodes_feature = torch.tensor([0 for _ in range(node_num)]).long().reshape(-1, 1)
        indices_src, indices_dst = edges
        if self.graph is None:
            self.graph = dgl.graph((indices_src, indices_dst))
            # self.graph.add_nodes(node_num)
            self.graph.ndata['feat'] = nodes
            self.graph.edata['feat'] = edge_attr.reshape(-1)
        G = self.graph

        if self.lap:
            G = lpe(G, pos_enc_dim=self.pos_enc)
        if self.wl_pos:
            G = wpe(G)
        if self.full:
            G = mfg(G)

        batch_x = G.ndata['feat']
        batch_e = G.edata['feat']
        try:
            batch_lap_pos_enc = G.ndata['lap_pos_enc']
            sign_flip = torch.rand(batch_lap_pos_enc.size(1))
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None

        try:
            batch_wl_pos_enc = G.ndata['wl_pos_enc']
        except:
            batch_wl_pos_enc = None



        # 2. Transform G with graph transformer to G_out
        out = self.gnn(G, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)  # out is the nodes coordinates

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



