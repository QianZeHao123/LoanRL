import pandas as pd 
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence 
from padsequence import *
from customized_dataloader import *
from multitask_lstm import *
from multitask_gru import *
from multitask_rnn import *
from masked_mse import *
from tqdm import tqdm





#parser = ArgumentParser()
#parser.add_argument('--input_size',"-is", type=int, required=True)
#parser.add_argument('--lr', '-l', type=float, required=True)
#parser.add_argument('--batch_size', '-bs', type=int, required=True)
#parser.add_argument('--hidden_size', '-hs', type=int, required=True)
#parser.add_argument('--alpha', '-a', type=float, required=True)
#parser.add_argument('--epoch', '-e', type=int, required=True)
#parser.add_argument('--training_data_path', '-dp', type=str, required=True)
#parser.add_argument('--which_model', '-wm', type=str, required=True)

'''
	- input_size: the input size of the simulator
	- lr : The learning rate
	- batch_size : The size of the batch when sample from memory experience
	- hidden_size : the hidden size of the LSTM/GRU/RNN
	- alpha: the coefficient of the state prediction task for multitask training
	- epoch: number of training epochs
	- training_data_path: training data absolute path
    -- which_model: select the base model, options are 'LSTM', 'GRU', 'RNN'

'''


#args = parser.parse_args()

data_path = 'Simulator/simulator_train_batch.pkl' #args.training_data_path
pkl_file = open(data_path, 'rb')
train_batch_data = pickle.load(pkl_file)
pkl_file.close()

ds = Mydataset(train_batch_data)

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 21 #args.input_size
LR = 0.001 #args.lr     
batch_size = 64 #args.batch_size
hidden_size = 50 #args.hidden_size
alpha = 0.25 #args.alpha
epochs = 100 #args.epoch
which_model = 'RNN' #args.which_model

train_loader =DataLoader(
    dataset = ds,
    batch_size = batch_size,
    shuffle = True,
    collate_fn=PadSequence(),
    num_workers = 2)

varstate_size = 7
model_zoo = {'LSTM': MultitaskLSTM(input_size=input_size, hidden_size=hidden_size, varstate_size=varstate_size),
             'GRU': MultitaskGRU(input_size=input_size, hidden_size=hidden_size, varstate_size=varstate_size),
             'RNN': MultitaskRNN(input_size=input_size, hidden_size=hidden_size, varstate_size=varstate_size)}

model = model_zoo[which_model]

optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = MaskedMSE()
loss_func_bc = MaskedBCE()
#print (model)


h_state = None
train_loss_record = []


for epoch in tqdm(range(epochs), desc=which_model):
    for step, (b_x, lenghts, b_y) in enumerate(train_loader): 
        state_prediction, installment_done_prediction, loan_done_prediction, reward_prediction, _ = model(b_x, lenghts,h_state)   
        
        state_mask = torch.zeros(state_prediction.size())
        installment_done_mask = torch.zeros(installment_done_prediction.size())
        loan_done_mask = torch.zeros(loan_done_prediction.size())
        reward_mask = torch.zeros(reward_prediction.size())

        for batch_id, item in enumerate(lenghts):
            state_mask[batch_id][:item]=1
            installment_done_mask[batch_id][:item]=1
            loan_done_mask[batch_id][:item]=1
            reward_mask[batch_id][:item]=1

        # state_mask = state_mask.type(torch.uint8)
        state_mask = state_mask.bool()
        installment_done_mask = installment_done_mask.bool()
        loan_done_mask = loan_done_mask.bool()
        # reward_mask = reward_mask.type(torch.uint8)
        reward_mask = reward_mask.bool()

        state_loss = loss_func(state_prediction, b_y[:,:,:varstate_size],state_mask )         # calculate loss
        installment_done_loss = loss_func_bc(installment_done_prediction, b_y[:,:,-3].unsqueeze(2), installment_done_mask)
        loan_done_loss = loss_func_bc(loan_done_prediction, b_y[:,:,-2].unsqueeze(2), loan_done_mask)
        reward_loss = loss_func(reward_prediction, b_y[:,:,-1].unsqueeze(2),reward_mask)

        #multi-task loss controlled by the factor alpha
        # loss = alpha * state_loss + (1-alpha) * reward_loss
        loss = alpha * state_loss + alpha * reward_loss + alpha * installment_done_loss + (1 - 3 * alpha) * loan_done_loss 

        optimizer.zero_grad()                   # clear gradients for this training step
        loss.backward()                         # backpropagation, compute gradients
        optimizer.step() 

        train_loss_record.append(loss.item())

        if step % 100 ==0:
            print (epoch, step, loss.item())
                
    
    torch.save(model, 'Simulator/generated_multitask_'+which_model+'_simulator_'+str(LR)+'_'+str(batch_size)+'.pkl')

    plt.plot(train_loss_record)
    plt.ylabel('training loss')
    #plt.show()
    plt.savefig('Simulator/Figure/training_loss_'+which_model+'_'+str(LR)+'_'+str(batch_size)+'.png')