import numpy as np 
import torch 
from torch_geometric.data import Data 
from torch_sparse import SparseTensor
import networkx as nx
from scipy.linalg import orthogonal_procrustes
import pdb


def eig_coord_from_dist(D):
    M = (D[:1, :] + D[:, :1] - D) / 2   # D: [B, N, N]  # N is the number of nodes
    L, V = torch.linalg.eigh(M)   # L= S, v = U
    L = torch.diag_embed(L)
    X = torch.matmul(V, L.clamp(min=0).sqrt())
    return X[:, -3:].detach()


def convert_to_matrix(adj):   # adj is the three columns tablet about Hi-C contact data
  temp1 = adj[:,0]
  temp2 = adj[:,1]
  temp3 = np.concatenate((temp1, temp2))
  idx = np.unique(temp3)
  size = len(idx)
  mat = np.zeros((size, size))
  for k in range(len(adj)):
    i = int(np.argwhere(adj[k, 0] == idx))
    j = int(np.argwhere(adj[k, 1] == idx))
    mat[i, j] = adj[k,2]
  mat = np.triu(mat) + np.tril(mat.T, 1)
  idx = np.argwhere(np.all(mat[..., :] == 0, axis=0))
  mat = np.delete(mat, idx, axis = 1)
  mat = np.delete(mat , idx, axis = 0)

  return mat


def load_input(input, features):
  adj_mat = input
  if adj_mat.shape[1] == 3:
    adj_mat = convert_to_matrix(adj_mat)
  np.fill_diagonal(adj_mat,0)
  truth = adj_mat
  truth = torch.tensor(truth,dtype=torch.double)
  graph = nx.from_numpy_array(adj_mat).to_undirected()  # this is the new version, for old version networkx, nx.from_numpy_matrix().to_undirected()
  num_nodes = adj_mat.shape[0]
  edges = list(graph.edges(data=True))
  edge_index = np.empty((len(edges),2))
  edge_weights = np.empty((len(edges)))
  nodes = np.empty(num_nodes)

  for i in range(len(edges)):
    edge_index[i] = np.asarray(edges[i][0:2])
    edge_weights[i] = np.asarray(edges[i][2]["weight"])

  for i in range(num_nodes):
    nodes[i] = np.asarray(i)

  edge_index = torch.tensor(edge_index, dtype=torch.long)
  edge_weights = torch.tensor(edge_weights, dtype=torch.float)
  nodes = torch.tensor(nodes, dtype=torch.long)
  node_attr = torch.tensor(features, dtype=torch.float)

  edge_index=edge_index.t().contiguous()

  
  mask = edge_index[0] != edge_index[1]  # in order to filter the diagonal index in the adj-matrix
  #inv_mask = ~mask

  edge_index = edge_index[:, mask]  # after filetring the diagonal, the new edge_index.
  edge_attr = edge_weights[mask]  # after filtering the diagonal, the new edge_weights.

  data = Data(x=node_attr,edge_index = edge_index, edge_attr = edge_attr, y = truth)

  #device = torch.device('cuda:0')
  #data = data.to(device)
 
  adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1] , value= data.edge_attr, sparse_sizes=(num_nodes, num_nodes))
  data.edge_index = adj.to_symmetric()
  data.edge_attr = None
  return data

edges_dic = {}
def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges

def load_info(normed, embeding, device):
    truth = normed
    truth = torch.tensor(truth, dtype=torch.double)

    # adj = cont2dist(truth, 0.5)
    # adj = adj.detach().to('cpu').numpy()
    # graph = nx.from_numpy_array(adj).to_directed()

    graph = nx.from_numpy_array(normed).to_directed()  # old version is nx.from_numpy_matrix().to_directed()
    num_nodes = normed.shape[0]
    node_attr = torch.tensor(embeding, dtype=torch.float)

    edges = list(graph.edges(data=True))
    edge_index = np.empty((len(edges), 2))
    edge_weights = np.empty((len(edges)))
    for i in range(len(edges)):
        edge_index[i] = np.asarray(edges[i][0:2])
        edge_weights[i] = np.asarray(edges[i][2]["weight"])

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    edge_index = edge_index.t().contiguous()
    mask = edge_index[0] != edge_index[1]   # mask is also a tensor, each element is 'Ttue' or 'False'
    mask = mask[:, None]

    # edge_index = edge_index[:, mask]  # after filetring the diagonal, the new edge_index.
    # edge_attr = edge_weights[mask]  # after filtering the diagonal, the new edge_weights.
    edge_attr = edge_weights.unsqueeze(-1)  # same function as edge_weights[:, None] that can extend the tensor to two dimension along last axis in order to concate,
    edge_ind = [edge_index[0].to(device), edge_index[1].to(device)]

    return node_attr.to(device), edge_ind, mask.to(device), edge_attr.to(device), truth.to(device), num_nodes


def load_UniMP(normed, embeding, device):
    truth = normed
    truth = torch.tensor(truth, dtype=torch.double)

    # graph = nx.from_numpy_array(normed).to_directed()  # old version is nx.from_numpy_matrix().to_directed()
    # num_nodes = normed.shape[0]
    node_attr = torch.tensor(embeding, dtype=torch.float)[None, :, :]

    edges = torch.tensor(normed, dtype=torch.float)
    edges = edges[None, :, :, None]

    adj = truth.to().bool().to().float()[None, :, :]
    return node_attr.to(device), edges.to(device), adj.to(device), truth.to(device)


def cont2dist(adj,factor):  # adj should be a tensor
  dist = (1/adj)**factor
  dist.fill_diagonal_(0)
  max = torch.max(torch.nan_to_num(dist,posinf=0))
  dist = torch.nan_to_num(dist,posinf=0)
  return dist/max


def Process_noise(data, noise_level = 'low'):

    normed = data
    norm = torch.tensor(normed, dtype=torch.float)

    # get the up triangle indices to add the noise in the form of N(mean, std)
    ind = torch.triu_indices(norm.shape[0], norm.shape[1], offset=1)
    val = norm[ind[0], ind[1]]   # find out the corresponding values in norm
    mask = val != 0  # mask all the indices of zero values
    ind = ind[:, mask]
    val = val[mask]  # get all the non-zero values
    sparse_clean = torch.cat((ind, val[None, :]), dim = 0).t().numpy()

    m = val.min()  # the smallest value as the mean for the low noisy data
    std = m

    if noise_level == 'middle' or noise_level == 'mid':
        m = 2*m
    elif noise_level == 'high':
        m = 3*m
    elif noise_level == 'low':
        m = 1*m
    else:
        raise Exception('Invalid noise level.')
        sys.exit(2)

    noise = torch.normal(m, std, size=val.shape)
    norm[ind[0], ind[1]] += noise  # add noise to corresponding entries which only within the up triangle
    val_noise = norm[ind[0], ind[1]]  # get the corresponding noisy entries

    norm_up = torch.triu(norm, diagonal=1)
    norm_low = norm_up.t()

    norm_noise = norm_up + norm_low   # generate the entire noisy normed square matrix.
    norm_noise = norm_noise.numpy()
    sparse_noise = torch.cat((ind, val_noise[None, :]), dim = 0).t().numpy()
    print(f'\nThe sparse clean data shape: {sparse_clean.shape} and its noisy data shape: {sparse_noise.shape}')
    print(f'\nThe square noisy data shape: {norm_noise.shape} and its non zero up triangle entries shape: {val_noise.shape}')

    assert sparse_clean.shape == sparse_noise.shape, "The clean and corresponding noisy data should have the same shape"

    return sparse_clean, sparse_noise, norm_noise


def cont2dist_sage(adj,factor):  # adj should be a tensor
  dist = (1/adj)**factor
  dist.fill_diagonal_(0)
  max = torch.max(torch.nan_to_num(dist,posinf=0))
  dist = torch.nan_to_num(dist,posinf=max)
  return dist/max

def domain_alignment(list1, list2, embeddings1, embeddings2):
  idx1 = np.unique(list1[:,0]).astype(int)
  diff1 = min(idx1[1:] - idx1[:-1])

  idx2 = np.unique(list2[:,0]).astype(int)
  diff2 = min(idx2[1:] - idx2[:-1])
  
  bins = (diff1/(2*diff2)).astype(int)

  A_list = []
  B_list = []

  for i in range(bins+1):
    Aidx = np.where(np.isin(idx2 + i*diff2, idx1))[0]
    Bidx = np.where(np.isin(idx1, idx2 + i*diff2))[0]

    A_list.append(embeddings2[Aidx,:])
    B_list.append(embeddings1[Bidx,:])


  A = np.concatenate(tuple(A_list))
  B = np.concatenate(tuple(B_list))

  transform = orthogonal_procrustes(A, B)[0]
  fitembed = np.matmul(embeddings2, transform)

  return fitembed

def WritePDB(positions, pdb_file, ctype = "0"):
  '''Save the result as a .pdb file'''
  o_file = open(pdb_file, "w")
  o_file.write("\n")

  col1 = "ATOM"
  col3 = "CA MET"
  col8 = "0.20 10.00"

  bin_num = len(positions)

  for i in range(1, bin_num+1):
      col2 = str(i)
      col4 = "B"+col2
      col5 = "%.3f" % positions[i-1][0]
      col6 = "%.3f" % positions[i-1][1]
      col7 = "%.3f" % positions[i-1][2]
      col2 = " "*(5 - len(col2)) + col2
      col4 = col4 + " " * (6 - len(col4))
      col5 = " " * (8 - len(col5)) + col5
      col6 = " " * (8 - len(col6)) + col6
      col7 = " " * (8 - len(col7)) + col7

      col = (col1, col2, col3, col4, col5, col6, col7,col8)
      line = "%s  %s   %s %s   %s%s%s  %s\n" % col
      o_file.write(line)
  col1 = "CONECT"
  for i in range(1, bin_num+1):
      col2 = str(i)
      j = i + 1
      if j > bin_num:
          if ctype == "1":
              continue
          #j = 1
      col3 = str(j)

      col2 = " " * (5 - len(col2)) + col2
      col3 = " " * (5 - len(col3)) + col3

      line = "%s%s%s\n" % (col1, col2, col3)
      o_file.write(line)

  o_file.write("END")
  o_file.close()





