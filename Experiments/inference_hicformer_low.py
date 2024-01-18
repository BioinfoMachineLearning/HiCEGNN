import os
import sys
import matplotlib.pyplot as plt

sys.path.append('../')
import subprocess

import torch
from torch import optim
from torch.nn import MSELoss
from torch import cdist
from tqdm import tqdm
import argparse

import ast
import numpy as np
from scipy.stats import spearmanr

from models.clof import ClofNet, ClofNet_vel, ClofNet_vel_gbf  # baseline models' modules
from Utils import utils  # pre-process and post-process functions
from Utils.utils import eig_coord_from_dist as coord

import wandb  # the logger
from ProcessData.DataProcess_new import HiC_Loader, HiC_noise_Loader  # the datasets
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

root_n = '/bmlfast/yw7bh/HiCFormer'

if not (os.path.exists(str(root) + '/Outputs')):
    subprocess.run("mkdir -p " + str(root) + '/Outputs', shell=True)

parser = argparse.ArgumentParser(description='Train the HiCFormer with different parameters.')

# below is about the parameters of optimizer
parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-16,
                    help='clamp the output of the coords function if get too large')

# initialize the EGNN model with the following settings.
# note: clof_human1_32res_*eps is trained by ClofNet_vel. clof_human1_*eps is trained by ClofNet
parser.add_argument('--mod_name', type=str, default='clof_human1_4res_1500eps', help='choice the model you want to use')
parser.add_argument('--epochs', type=int, default=1501, help='number of epochs to train (default: 10)')  # default = 100
parser.add_argument('--n_layers', type=int, default=4, help='number of layers for the EGLN')  # test 3 is better than 4, 4 is better than 6
parser.add_argument('--attention', type=int, default=1, help='attention in the EGNN model')
parser.add_argument('--node_dim', type=int, default=512, help='Number of node features at input')
parser.add_argument('--edge_dim', type=int, default=1, help='Number of edge features at input')
parser.add_argument('--nf', type=int, default=128, help='Number of hidden features')  # default 128
parser.add_argument('--node_attr', type=int, default=1, help='node_attr or not')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',  help='use tanh')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N', help='normalize_diff')

# set the device to train the model
parser.add_argument('-c', '--conversions', type=str, default='[.5]', help='List of conversion constants of the form [lowest, interval, highest] for an set of\
        equally spaced conversion factors, or of the form [conversion] for a single conversion factor.')  # '[.1,.1, 2.1]'
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

conversions = args.conversions
conversions = ast.literal_eval(conversions)

if len(conversions) == 3:
    conversions = list(np.arange(conversions[0], conversions[2], conversions[1]))
elif len(conversions) == 1:
    conversions = [conversions[0]]
else:
    raise Exception('Invalid conversion input.')
    sys.exit(2)


class HiCFormer:
    def __init__(self, cell_Line='Human', cellNo=10, res=40000, model=args.mod_name, batch_size=1):
        # initialize the parameters that will be used during fit model
        self.cell_Line = cell_Line
        self.cellNo = cellNo
        self.res = res

        # whether using GPU for training
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        self.device = device
        if self.cellNo == 10 or self.cellNo == 21 or self.cellNo == 22:
            self.mod_name = 'clof_4ReLU_400eps'
        else:
            self.mod_name = model

        # out_dir: directory storing checkpoint files and parameters for saving to the our_dir
        dir_name = 'Model_Weights'
        self.out_dir = os.path.join(root_n, dir_name)
        # self.out_dirM = os.path.join(root, "Metrics")
        os.makedirs(self.out_dir, exist_ok=True)  # makedirs will make all the directories on the path if not exist.
        # os.makedirs(self.out_dirM, exist_ok=True)

        self.valid_loader = HiC_Loader(full=True,
                                       tvt='test',
                                       res=res,
                                       celline=cell_Line,
                                       cellno=self.cellNo,
                                       epoch=5,
                                       lstm=False,
                                       shuff=False)

        # initialize the model for training, the default parameters for attention in gcl is false
        if self.cellNo == 10 or self.cellNo == 21 or self.cellNo == 22:
            self.model = ClofNet(in_node_nf=args.node_dim, in_edge_nf=args.edge_dim, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
        else:
            self.model = ClofNet_vel(in_node_nf=args.node_dim, in_edge_nf=args.edge_dim, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)

    def fit_model(self):
        # optimizer

        tempspear = []
        tempmse = []
        dt = 0.05
        for conversion in conversions:
            if self.cellNo == 10 or self.cellNo == 21 or self.cellNo == 22:
                best_ckpt_file = 'bestg_500000_c0.5_Human_hicclof_4ReLU_400eps.pytorch'
            else:
                best_ckpt_file = f'bestg_{self.res}_c{conversion}_Human_1_hic{self.mod_name}.pytorch'
            print(best_ckpt_file)
            path = os.path.join(self.out_dir, best_ckpt_file)
            self.model.load_state_dict(torch.load(path))

            print(f'Training model using conversion value {conversion}.')
            criterion = MSELoss()
            valid_result = {'nsamples': 0, 'loss': 0, 'dscc': 0}
            self.model.eval()
            valid_bar = tqdm(self.valid_loader)

            with torch.no_grad():
                for data, target, info in valid_bar:  # data is the pure image without noise
                    # if info == 18:
                        # continue
                    data, = data  # used to unpack the only one item in the object such as tuple or list
                    target, = target  # used to unpack the only one item in the object such as tuple or list

                    normed = np.loadtxt(data)
                    embeding = np.loadtxt(target)
                    node_attr, edge_ind, mask, edge_attr, truth, num_nodes = utils.load_info(normed, embeding,
                                                                                                 device=device)
                    truth = utils.cont2dist(truth, .5)  # the wish distance map as the target
                    normed = torch.tensor(normed, dtype=torch.float).to(device)

                    # print(f'\n********** the number of chrom {info} node {num_nodes} **************')
                    truth_n = torch.tensor(truth, dtype=torch.float).to(device)
                    D = truth_n * truth_n
                    x_D = coord(D)
                    vel = torch.zeros_like(x_D).to(device)

                    batch_size = len(info)
                    valid_result['nsamples'] += batch_size


                    h, out, dist = self.model(h=node_attr, x = x_D, edges=edge_ind,  vel = vel,  edge_attr=edge_attr, n_nodes=num_nodes)

                    norm1 = torch.tensor(normed, dtype=torch.float).to(device)
                    dices = torch.where(norm1 != 0.0)

                    loss = criterion(dist.float()[dices], truth.float()[dices])
                    valid_result['loss'] += loss.item() * batch_size

                    # idx = torch.triu_indices(truth.shape[0], truth.shape[1], offset=1)
                    # dist_truth = truth[idx[0, :], idx[1, :]]
                    # dist_out = dist[idx[0, :], idx[1, :]]dist_truth = truth[dices]

                    # dist = cdist(x_D, x_D, p=2)
                    dist_truth = truth[dices]
                    dist_out = dist[dices]

                    # print(dist_truth.device)
                    dscc = spearmanr(dist_truth.cpu(), dist_out.detach().cpu().numpy())[0]
                    valid_result['dscc'] += dscc * batch_size
                    # print(f'------------ the chrom is {info} and the loss is {loss} and dscc is {dscc} ---------------\n')

                valid_loss = valid_result['loss'] / valid_result['nsamples']
                valid_dscc = valid_result['dscc'] / valid_result['nsamples']
                dscc_lits = valid_result['dscc']
                # print(f'------------ the epoch is {epoch} and the dscc is {valid_dscc} ---------------\n')

            repconv = conversion
            repspear = valid_dscc
            repmse = valid_loss
            print(f'Optimal conversion factor: {repconv}')
            print(f'Optimal dSCC: {repspear}')

            with open(str(root) + '/Outputs/' + f'test_hic{self.mod_name}_{self.res}_{self.cell_Line}_{self.cellNo}_log.txt', 'w') as f:
                line2 = f'Optimal dSCC: {repspear}\n'
                line3 = f'Final MSE loss: {repmse}\n'
                f.writelines([line2, line3])


if __name__ == "__main__":
    train_model = HiCFormer(cell_Line='Dros', cellNo=22, res=320000)
    train_model.fit_model()
    print("\n\nTraining clofnet is done!!!\n")
