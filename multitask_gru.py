import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


class MultitaskGRU(nn.Module):

    def __init__(self, input_size, hidden_size, varstate_size):
        super(MultitaskGRU, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.state_layer = nn.Linear(hidden_size, varstate_size)
        self.installment_done_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.loan_done_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.reward_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )

    def forward(self, x, x_lengths, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        x = pack_padded_sequence(x, x_lengths, batch_first=True)
        r_out, h_state = self.gru(x, h_state)
        paded_out, paded_lengths = pad_packed_sequence(r_out, batch_first=True)

        state_outs = []
        installment_done_outs = []
        loan_done_outs = []
        reward_outs = []
        # calculate output for each time step
        for time_step in range(paded_out.size(1)):
            state_outs.append(self.state_layer(paded_out[:, time_step, :]))
            installment_done_outs.append(
                self.installment_done_layer(paded_out[:, time_step, :]))
            loan_done_outs.append(
                self.loan_done_layer(paded_out[:, time_step, :]))
            reward_outs.append(self.reward_layer(paded_out[:, time_step, :]))
        return torch.stack(state_outs, dim=1), torch.stack(installment_done_outs, dim=1), torch.stack(loan_done_outs, dim=1), torch.stack(reward_outs, dim=1), h_state
