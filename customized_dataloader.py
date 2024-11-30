import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class Mydataset(Dataset):
    def __init__(self, batch_data):
        self.data = batch_data
    
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = self.data[idx][:,1:22]
        y = self.data[idx][:,22:]

        return (X,y)