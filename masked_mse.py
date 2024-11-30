import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MaskedMSE(nn.Module):
    def __init__(self):
        super(MaskedMSE, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, prediction, target, mask):
        prediction = torch.masked_select(prediction, mask)
        target = torch.masked_select(target, mask)
        self.loss = self.criterion(prediction, target)
        return self.loss


class MaskedBCE(nn.Module):
    def __init__(self):
        super(MaskedBCE, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, prediction, target, mask):
        prediction = torch.masked_select(prediction, mask)
        target = torch.masked_select(target, mask)
        self.loss = self.criterion(prediction, target)
        return self.loss
