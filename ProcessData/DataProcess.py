import sys
sys.path.append('../')

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import os
import math
import subprocess
import glob
import numpy as np
import torch
# from ge import LINE
import networkx as nx
from Utils import utils

import gc
import pyrootutils
import cooler

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


'''
class HiCdata(Dataset):
    def __init__(self,
                 full,
                 tvt,
                 res = 40000,
                 cell_line = 'Human',
                 cell_No = 10,
                 epochs = 5,
                 batchsize = 128):

        self.res = res
        self.cellLine = cell_line
        self.cellNo = cell_No
        self.epoch = epochs
        self.batchsize = batchsize

        self.dirname = str(root) + "/Data" + "/DataFull_" + self.cellLine + "_cell" + str(self.cellNo) + "_" + str(self.res)

        self.target = []
        self.data = []
        self.info = []

        globs = glob.glob(self.dirname + "/Constraints/chrom_*_" + 'count' + ".txt")
        if len(globs) == 0:
            print("wait.. first we need to extract mats and double check the mats")
            #  input("Press Enter to continue...")
            self.extract_constraint_mats()

        globs = glob.glob(self.dirname + "/Constraints/chrom_*_" + 'count' +'_embeddings.txt')
        if len(globs) == 0:
            print("wait.. first we need to extract mats and double check the mats")
            #  input("Press Enter to continue...")

        if full == True:
            if tvt in list(range(1, 23)):
                self.chros = [tvt]
            if tvt == "train":
                self.chros = [1, 3, 5, 7, 8, 9, 11, 13, 15, 16, 17, 19, 21, 22]
            elif tvt == "val":
                self.chros = [4, 14, 18, 20]  #
            elif tvt == "test":
                self.chros = [2, 6, 10, 12]  # original is [2, 6, 10, 12]

            for c, chro in enumerate(self.chros):
                outdir = self.dirname + "/Constraints"
                filepath = outdir + '/chrom_' + str(chro) + '_' + 'count_500000' + '.txt'
                name = os.path.splitext(os.path.basename(filepath))[0]
                normed = outdir + '/' + name + '_matrix_KR_normed.txt'
                self.data.append(normed)

                embeded = outdir + '/' + name + '_embeddings.txt'
                self.target.append(embeded)
                self.info.append(chro)
        else:
            if tvt == "train":
                self.chros = [15]
            elif tvt == "val":
                self.chros = [16]
            elif tvt == "test":
                self.chros = [17]

            for c, chro in enumerate(self.chros):
                outdir = self.dirname + "/Constraints"
                filepath = outdir + '/chrom_' + str(chro) + '_' + 'count_500000' + '.txt'
                name = os.path.splitext(os.path.basename(filepath))[0]
                normed = outdir + '/' + name + '_matrix_KR_normed.txt'
                self.data.append(normed)

                embeded = outdir + '/' + name + '_embeddings.txt'
                self.target.append(embeded)
                self.info.append(chro)

    def extract_constraint_mats(self):
        if not os.path.exists(self.dirname+"/Constraints"):
            subprocess.run("mkdir -p "+self.dirname+"/Constraints", shell = True)

        outdir = self.dirname+"/Constraints"
        file_inter = glob.glob(str(root) + '/Datasets/Human/single/' + 'cell'+str(self.cellNo)+'_'+r'*.mcool')
        filepath = file_inter[0]
        print(f'------------the file_path for the cellline if {file_inter}')
        AllRes = cooler.fileops.list_coolers(filepath)
        print(AllRes)

        c = cooler.Cooler(filepath + '::resolutions/' + str(self.res))
        c1 = c.chroms()[:]  # c1 chromesize information in the list format
        print(c1.loc[0, 'name'], c1.index)
        # print('\n')

        for i in c1.index:
            print(i, c1.loc[i, 'name'])
            chro = c1.loc[i, 'name']  # chro is str
            # print(type(chro))
            c2 = c.matrix(balance = True, as_pixels = True, join = True).fetch(chro)
            c3 = c2[['start1', 'start2', 'count']]
            # print(c2)
            c2 = c2[['start1', 'start2', 'balanced']]
            c2.fillna(0, inplace = True)
            # print(c2)
            if i >= 22:
                pass
            else:
                c2.to_csv(outdir+'/chrom_' + str(i + 1) + '_' + str(self.res) + '.txt', sep = '\t', index = False, header = False) # balanced
                c3.to_csv(outdir+'/chrom_' + str(i + 1) + '_' + 'count' + '.txt', sep = '\t', index = False, header = False)  # raw count


    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx], self.info[idx]
'''

class HiCdata_n(Dataset):
    def __init__(self,
                 full,
                 tvt,
                 res = 40000,
                 cell_line = 'Human',
                 cell_No = 10,
                 epochs = 5,
                 lstm = True,
                 batchsize = 128):

        self.res = res
        self.cellLine = cell_line
        self.cellNo = cell_No
        self.epoch = epochs
        self.batchsize = batchsize
        self.lstm = lstm

        self.dirname = str(root) + "/Data/GSE130711"
        self.dirname_ful = str(root) + "/DataFull/GSE130711_" + cell_line + "_cell" + str(cell_No)

        self.target = []
        self.data = []
        self.info = []

        if self.cellNo == 10 and self.cellLine == "Human":
            inter_dir = self.dirname + "/Inputs"
        else:
            inter_dir = self.dirname + "/" + self.cellLine + "_cell" + str(self.cellNo)

        # print(inter_dir)
        if self.lstm:
            globs = glob.glob(inter_dir + "/chrom_*_" + str(self.res) + "_lstm_embeddings" + ".txt")
        else:
            globs = glob.glob(inter_dir + "/chrom_*_" + str(self.res) + "_embeddings" + ".txt")
        # print(globs)
        if len(globs) == 0:
            print("wait.. first we need to extract mats and double check the mats")
            #  input("Press Enter to continue...")
            self.extract_constraint_mats()

        if full == True:
            if tvt in list(range(1, 23)):
                self.chros = [tvt]
            if tvt == "train":
                self.chros = [1, 3, 5, 7, 8, 9, 11, 13, 15, 16, 17, 19, 21, 22]
            elif tvt == "val":
                self.chros = [4, 14, 18, 20]  #
            elif tvt == "test":
                self.chros = [2, 6, 10, 12]  # original is [2, 6, 10, 12]

            for c, chro in enumerate(self.chros):
                if self.cellNo == 10 and self.cellLine == "Human":
                    outdir = self.dirname + "/Inputs"
                else:
                    outdir = self.dirname + "/" + self.cellLine + "_cell" + str(self.cellNo)

                # print(outdir)
                filepath = outdir + '/chrom_' + str(chro) + '_' + str(self.res) + '.txt'
                name = os.path.splitext(os.path.basename(filepath))[0]
                normed = outdir + '/' + name + '_matrix_KR_normed.txt'
                self.data.append(normed)

                if self.lstm:
                    embeded = outdir + '/' + name + '_lstm_embeddings.txt'
                else:
                    embeded = outdir + '/' + name + '_embeddings.txt'
                self.target.append(embeded)
                self.info.append(chro)
        else:
            if tvt == "train":
                self.chros = [15]
            elif tvt == "val":
                self.chros = [16]
            elif tvt == "test":
                self.chros = [17]

            for c, chro in enumerate(self.chros):
                # outdir = None
                if self.cellNo == 10 and self.cellLine == "Human":
                    outdir = self.dirname + "/Inputs"
                else:
                    outdir = self.dirname + "/" + self.cellLine + "_cell" + str(self.cellNo)

                # print(outdir)
                filepath = outdir + '/chrom_' + str(chro) + '_' + str(self.res) + '.txt'
                name = os.path.splitext(os.path.basename(filepath))[0]
                normed = outdir + '/' + name + '_matrix_KR_normed.txt'
                self.data.append(normed)

                if self.lstm:
                    embeded = outdir + '/' + name + '_lstm_embeddings.txt'
                else:
                    embeded = outdir + '/' + name + '_embeddings.txt'
                self.target.append(embeded)
                self.info.append(chro)

    def extract_constraint_mats(self):
        if not os.path.exists(self.dirname_ful+"/Constraints"):
            subprocess.run("mkdir -p "+self.dirname_ful+"/Constraints", shell = True)

        outdir = self.dirname_ful+"/Constraints"
        file_inter = glob.glob(str(root) + '/Datasets/Human/single/' + 'cell'+str(self.cellNo)+'_'+r'*.mcool')
        filepath = file_inter[0]
        print(f'------------the file_path for the cellline is {file_inter}')
        AllRes = cooler.fileops.list_coolers(filepath)
        print(AllRes)

        c = cooler.Cooler(filepath + '::resolutions/' + str(self.res))
        c1 = c.chroms()[:]  # c1 chromesize information in the list format
        print(c1.loc[0, 'name'], c1.index)
        # print('\n')

        for i in c1.index:
            print(i, c1.loc[i, 'name'])
            chro = c1.loc[i, 'name']  # chro is str
            # print(type(chro))
            c2 = c.matrix(balance = True, as_pixels = True, join = True).fetch(chro)
            c3 = c2[['start1', 'start2', 'count']]
            # print(c2)
            c2 = c2[['start1', 'start2', 'balanced']]
            c2.fillna(0, inplace = True)
            # print(c2)
            if i >= 22:
                pass
            else:
                c2.to_csv(outdir+'/chrom_' + str(i + 1) + '_' + 'ble_'+  str(self.res) + '.txt', sep = '\t', index = False, header = False) # balanced
                c3.to_csv(outdir+'/chrom_' + str(i + 1) + '_' + 'ct_' + str(self.res) + '.txt', sep = '\t', index = False, header = False)  # raw count


    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx], self.info[idx]


class HiCdata_noise(Dataset):
    def __init__(self,
                 full,
                 tvt,
                 res = 40000,
                 cell_line = 'Human',
                 cell_No = 10,
                 epochs = 5,
                 lstm = False,
                 noise_level = 'low',
                 batchsize = 128):

        self.res = res
        self.cellLine = cell_line
        self.cellNo = cell_No
        self.epoch = epochs
        self.batchsize = batchsize
        self.lstm = lstm
        self.noise= noise_level

        self.dirname = str(root) + "/Data/GSE130711"
        self.dirname_ful = str(root) + "/DataFull/GSE130711_" + cell_line + "_cell" + str(cell_No)

        self.target = []  # for embeddings
        self.data = []   # for clean normed square matrix
        self.data_noise = [] # for noisy square matrix
        self.info = []

        if self.cellNo == 10 and self.cellLine == "Human":
            inter_dir = self.dirname + "/Inputs"
        else:
            inter_dir = self.dirname + "/" + self.cellLine + "_cell" + str(self.cellNo)

        # print(inter_dir)
        if self.lstm:
            globs = glob.glob(inter_dir + "/chrom_*_" + str(self.res) + "_lstm_embeddings_noise"+self.noise+ ".txt")
        else:
            globs = glob.glob(inter_dir + "/chrom_*_" + str(self.res) + "_embeddings_noise_"+self.noise + ".txt")
        # print(globs)
        if len(globs) == 0:
            print("wait.. first we need to extract mats and double check the mats")
            #  input("Press Enter to continue...")
            self.extract_constraint_mats()

        if full == True:
            if tvt in list(range(1, 23)):
                self.chros = [tvt]
            if tvt == "train":
                self.chros = [1, 3, 5, 7, 8, 9, 11, 13, 15, 16, 17, 19, 21, 22]
            elif tvt == "val":
                self.chros = [4, 14, 18, 20]  #
            elif tvt == "test":
                self.chros = [2, 6, 10, 12]  # original is [2, 6, 10, 12]

            for c, chro in enumerate(self.chros):
                if self.cellNo == 10 and self.cellLine == "Human":
                    outdir = self.dirname + "/Inputs"
                else:
                    outdir = self.dirname + "/" + self.cellLine + "_cell" + str(self.cellNo)

                # print(outdir)
                filepath = outdir + '/chrom_' + str(chro) + '_' + str(self.res) + '.txt'
                name = os.path.splitext(os.path.basename(filepath))[0]
                normed = outdir + '/' + name + '_matrix_KR_normed.txt'
                self.data.append(normed)

                noised = outdir + '/' + name + '_matrix_KR_normed_noise_'+self.noise+'.txt'
                self.data_noise.append(noised)

                if self.lstm:
                    embeded = outdir + '/' + name + '_lstm_embeddings_noise_'+self.noise+'.txt'
                else:
                    embeded = outdir + '/' + name + '_embeddings_noise_'+self.noise+'.txt'
                self.target.append(embeded)
                self.info.append(chro)
        else:
            if tvt == "train":
                self.chros = [15]
            elif tvt == "val":
                self.chros = [16]
            elif tvt == "test":
                self.chros = [17]

            for c, chro in enumerate(self.chros):
                # outdir = None
                if self.cellNo == 10 and self.cellLine == "Human":
                    outdir = self.dirname + "/Inputs"
                else:
                    outdir = self.dirname + "/" + self.cellLine + "_cell" + str(self.cellNo)

                # print(outdir)
                filepath = outdir + '/chrom_' + str(chro) + '_' + str(self.res) + '.txt'
                name = os.path.splitext(os.path.basename(filepath))[0]
                normed = outdir + '/' + name + '_matrix_KR_normed.txt'
                self.data.append(normed)

                noised = outdir + '/' + name + '_matrix_KR_normed_noise_'+self.noise+'.txt'
                self.data_noise.append(noised)

                if self.lstm:
                    embeded = outdir + '/' + name + '_lstm_embeddings_noise_'+self.noise+'.txt'
                else:
                    embeded = outdir + '/' + name + '_embeddings_noise_'+self.noise+'.txt'
                self.target.append(embeded)
                self.info.append(chro)

    def extract_constraint_mats(self):
        if not os.path.exists(self.dirname_ful+"/Constraints"):
            subprocess.run("mkdir -p "+self.dirname_ful+"/Constraints", shell = True)

        outdir = self.dirname_ful+"/Constraints"
        file_inter = glob.glob(str(root) + '/Datasets/Human/single/' + 'cell'+str(self.cellNo)+'_'+r'*.mcool')
        filepath = file_inter[0]
        print(f'------------the file_path for the cellline is {file_inter}')
        AllRes = cooler.fileops.list_coolers(filepath)
        print(AllRes)

        c = cooler.Cooler(filepath + '::resolutions/' + str(self.res))
        c1 = c.chroms()[:]  # c1 chromesize information in the list format
        print(c1.loc[0, 'name'], c1.index)
        # print('\n')

        for i in c1.index:
            print(i, c1.loc[i, 'name'])
            chro = c1.loc[i, 'name']  # chro is str
            # print(type(chro))
            c2 = c.matrix(balance = True, as_pixels = True, join = True).fetch(chro)
            c3 = c2[['start1', 'start2', 'count']]
            # print(c2)
            c2 = c2[['start1', 'start2', 'balanced']]
            c2.fillna(0, inplace = True)
            # print(c2)
            if i >= 22:
                pass
            else:
                c2.to_csv(outdir+'/chrom_' + str(i + 1) + '_' + 'ble_'+ str(self.res) + '.txt', sep = '\t', index = False, header = False) # balanced
                c3.to_csv(outdir+'/chrom_' + str(i + 1) + '_' + 'ct_' + str(self.res) + '.txt', sep = '\t', index = False, header = False)  # raw count


    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.data[idx], self.data_noise[idx], self.target[idx], self.info[idx]


def HiC_Loader(full = True,
                 tvt = None,
                 res = 500000,
                 celline = 'Human',
                 cellno = 10,
                 epoch = 5,
                 lstm = True,
                 shuff =  True):

    # if cellno == 10:  # for original data processing from Human cell 10 (Pseudo-bulk Hi-C data)
    dataset = HiCdata_n(full = full,
                 tvt = tvt,
                 res = res,
                 cell_line = celline,
                 cell_No = cellno,
                 epochs = epoch,
                 lstm=lstm)
    '''
    else:   # for new data processing
        dataset = HiCdata_n(full=full,
                          tvt=tvt,
                          res=res,
                          cell_line=celline,
                          cell_No=cellno,
                          epochs=epoch,
                          lstm = lstm)
    '''

    Loader = DataLoader(dataset, batch_size = 1, shuffle = shuff)
    return Loader



def HiC_noise_Loader(full = True,
                 tvt = None,
                 res = 500000,
                 celline = 'Human',
                 cellno = 10,
                 epoch = 5,
                 lstm = True,
                 noise_level = 'low',
                 shuff =  True):

    # for new data processing
    dataset = HiCdata_noise(full=full,
                        tvt=tvt,
                        res=res,
                        cell_line=celline,
                        cell_No=cellno,
                        epochs=epoch,
                        lstm = lstm,
                        noise_level=noise_level)


    Loader = DataLoader(dataset, batch_size = 1, shuffle = shuff)
    return Loader


if __name__ == "__main__":
    Loader = HiC_noise_Loader(cellno = 10, tvt='val', res=500000, lstm=False, noise_level='low')
    device = torch.device("cpu")
    i = 1
    for data, noise, target, _ in Loader:
        data, = data  # used to unpack the only one item in the object such as tuple or list
        noise, = noise
        target, = target  # used to unpack the only one item in the object such as tuple or list
        print(data)
        print(noise)
        print(target)
        print(i)
        i = i + 1

        '''
        normed = np.loadtxt(data)
        embeding = np.loadtxt(target)
        print(normed)
        print(target)
        node_attr, edge_ind, mask, edge_attr, truth, num_nodes = utils.load_info(normed, embeding, device=device)
        distance = utils.cont2dist(truth, 0.5)

        print(node_attr.shape)
        print(edge_ind[0].shape)
        print(mask.shape)
        print(edge_attr.shape)
        print(truth.shape)
        print(num_nodes)
        print(distance)
        print(distance.shape)
        '''

