import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class PadSequence:
    def __call__(self, batch):
        # import pdb; pdb.set_trace()

        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        # Get each sequence and pad it
        sequences = [torch.Tensor(x[0]) for x in sorted_batch]
        # print [item.size() for item in sequences]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        
        target = [torch.Tensor(x[1]) for x in sorted_batch]
        target_padded = torch.nn.utils.rnn.pad_sequence(target, batch_first=True)
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])
        return sequences_padded, lengths, target_padded
    