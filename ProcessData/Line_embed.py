import numpy as np
from ge import LINE
import argparse
import networkx as nx
# from Utils import utils
import os
import torch
import sys
sys.path.append('../')

#import pyrootutils
import glob

'''
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
'''
root = "/home/yw7bh/Projects/HiCFormer"


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

if __name__=="__main__":

        
    parser = argparse.ArgumentParser(description='Generate embeddings for an input file.')
    parser.add_argument('--filepath', type=str, help='Input text file path. The file should be either a tab-delimeted interaction frequency matrix, or a  tab-delimeted\
        coordinate list of the form [position1, position2, interaction_frequency].')
    parser.add_argument('-bs', '--batchsize',  type=int, default=512, help='Batch size for embeddings generation.')
    parser.add_argument('-ep', '--epochs', type=int, default=30, help='Number of epochs used for embeddings generation')
    parser.add_argument('-cl', '--celline', type=str, default='Dros', help='Number of epochs used for embeddings generation')
    parser.add_argument('-cn', '--cellno', type=int, default=22, help='Number of epochs used for embeddings generation')
    parser.add_argument('-re', '--resolution', type=int, default=320000, help='Number of epochs used for embeddings generation')
    parser.add_argument('-cp', '--clip', type=float, default=99.9, help='Number of epochs used for embeddings generation')
    parser.add_argument('-ni', '--noise', type=int, default=1, help='Whether add noise or not')
    parser.add_argument('-nl', '--noiselevel', type=str, default='low', help='Add noise to the clean data')
    args = parser.parse_args()
    # filepath = args.filepath
    batch_size = args.batchsize
    epochs = args.epochs

    if args.cellno == 10 or args.cellno == 21 or args.cellno == 22:
        if args.celline == "Human":
            file_o = root + '/Data/GSE130711/Inputs'
        else:
            file_o = root + '/Data/GSE131811/' + args.celline + "_cell" + str(args.cellno)
        if not (os.path.exists(file_o)):
            os.makedirs(file_o, exist_ok=True)

        args.clip = 99.0
    else:
        if args.celline == "Human":
            file_o = root + '/Data/GSE130711/'+args.celline+"_cell"+str(args.cellno)
        else:
            file_o = root + '/Data/GSE131811/' + args.celline + "_cell" + str(args.cellno)
        if not (os.path.exists(file_o)):
            os.makedirs(file_o, exist_ok=True)
        args.clip = 99.9

    if args.celline == "Human":
        dir = root+ "/DataFull/GSE130711_Human_cell"+str(args.cellno) + "/Full_Mats"
    else:
        dir = root + "/DataFull/GSE131811_Dros_cell" + str(args.cellno) + "/Full_Mats"
    globs = glob.glob(dir + "/chrom_*_full_ble_"+str(args.resolution)+".npy")

    for file in globs:
        filepath = file
        adj = np.load(filepath)
        np.fill_diagonal(adj, 0)
        per_a = np.percentile(adj, args.clip)
        adj = np.clip(adj, 0, per_a)
        adj = adj/per_a

        G = nx.from_numpy_array(adj)

        # if need to add noise, we should embed the noisy data from the normed_clean square matrix
        normed_noise = None
        sparse_clean = None
        sparse_noise = None
        if args.noise:
            sparse_clean, sparse_noise, normed_noise = Process_noise(adj, args.noiselevel)
            G = nx.from_numpy_array(normed_noise)

        embed = LINE(G,embedding_size=512,order='second')
        embed.train(batch_size=batch_size,epochs=epochs,verbose=1)
        embeddings = embed.get_embeddings()
        embeddings = list(embeddings.values())
        embeddings = np.asarray(embeddings)

        name = os.path.splitext(os.path.basename(filepath))[0].split('_')
        name = "/"+name[0]+"_"+name[1]+"_"+name[-1]

        if args.noise:
            np.savetxt(file_o + name + '_embeddings_noise_'+args.noiselevel+'.txt', embeddings)
            np.savetxt(file_o + name + '_matrix_KR_normed_noise_'+args.noiselevel+'.txt', normed_noise)
            np.savetxt(file_o + name + '_sparse_clean.txt', sparse_clean)
            np.savetxt(file_o + name + '_sparse_noise_'+args.noiselevel+'.txt', sparse_noise)
        else:
            # Orignal stored information:
            np.savetxt(file_o + name + '_embeddings.txt', embeddings)
            np.savetxt(file_o + name + '_matrix_KR_normed.txt', adj)

        print('Created embeddings corresponding to ' + filepath +  " as: "+ file_o + name + '_embeddings.txt')
