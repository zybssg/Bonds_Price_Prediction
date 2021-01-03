import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

class Dataset_Bonds(Dataset):
    def __init__(self, data):
        self.feats = data['feats']
        self.labels = data['labels']

    def __getitem__(self, index):
        feat = self.feats[index]
        label = self.labels[index]
        return feat, label

    def __len__(self):
        return len(self.feats)


# # cell test
# import sys
# sys.path.append('./')
# from utils.args_lstm import args
# print(args)
# mydataset = Dataset_Bonds(args)
