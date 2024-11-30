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
import pdb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error





#parser = ArgumentParser()
#parser.add_argument('--input_size',"-is", type=int, required=True)
#parser.add_argument('--lr', '-l', type=float, required=True)
#parser.add_argument('--batch_size', '-bs', type=int, required=True)
#parser.add_argument('--hidden_size', '-hs', type=int, required=True)
#parser.add_argument('--alpha', '-a', type=float, required=True)
#parser.add_argument('--epoch', '-e', type=int, required=True)
# parser.add_argument('--training_data_path', '-dp', type=str, required=True)
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

# pkl_file = open(args.training_data_path, 'rb')
# pkl_file = open('simulator_stratified_training_batch.pkl', 'rb')
# train_batch_data = pickle.load(pkl_file)
# pkl_file.close()

def cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # Small value to avoid log(0)
    N = len(y_true)  # Number of samples
    
    if not np.all(np.logical_or(y_true == 0, y_true == 1)):
        raise ValueError("y_true should only contain 0 and 1 values")
    
    if not np.all((0 <= y_pred) & (y_pred <= 1)):
        print('y_pred before clipping ', y_pred)
        raise ValueError("y_pred should contain values between 0 and 1")
    
    # Clip predicted values to avoid log(0): Surprisingly, this clip does not work for 1
    # y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    y_pred_clip = []
    for i in y_pred:
        if i < epsilon:
            y_pred_clip.append(epsilon)
        elif i > 1 - epsilon:
            y_pred_clip.append(1 - epsilon)
        else:
            y_pred_clip.append(i)
            
    y_pred_clip = np.array(y_pred_clip)
    
    if not np.all((0 < y_pred_clip) & (y_pred_clip < 1)):
        print('y_pred after clipping ', y_pred_clip)
        raise ValueError("after clipping, y_pred should contain values strictly between 0 and 1")
    
    # Calculate cross entropy
    ce = -np.sum(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip)) / N
    
    return ce

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

pkl_file = open('simulator_training_train_batch.pkl', 'rb')
train_data = pickle.load(pkl_file)
pkl_file.close()

valid_data = pd.read_csv('simulator_training_valid_batch.csv')

ds_train = Mydataset(train_data)
#ds_test = Mydataset(valid_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 20
LR_list = [0.001,0.005]    
# LR = 0.005 
batch_size_list = [16, 32, 64]
#batch_size_list = [16]
# hidden_size = 50
hidden_size_list = [50, 100, 150]
#hidden_size_list = [100]
# alpha = args.alpha
epochs = 100
which_model = 'LSTM'


if __name__ == "__main__":
    
    for LR in LR_list:
        for batch_size in batch_size_list:
            for hidden_size in hidden_size_list:
                
                train_loader =DataLoader(
                dataset = ds_train,
                batch_size = batch_size,
                shuffle = True,
                collate_fn=PadSequence(),
                num_workers = 2)

                model_zoo = {'LSTM': MultitaskLSTM(input_size=input_size, hidden_size=hidden_size),
                            'GRU': MultitaskGRU(input_size=input_size, hidden_size=hidden_size),
                            'RNN': MultitaskRNN(input_size=input_size, hidden_size=hidden_size)}

                model = model_zoo[which_model]

                optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
                loss_func = MaskedMSE()
                loss_func_bc = MaskedBCE()
                print (model)

                h_state = None
                train_loss_record = []
                for epoch in range(epochs):
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

                        state_loss = loss_func(state_prediction, b_y[:,:,:6],state_mask )         # calculate loss
                        installment_done_loss = loss_func_bc(installment_done_prediction, b_y[:,:,-3].unsqueeze(2), installment_done_mask)
                        loan_done_loss = loss_func_bc(loan_done_prediction, b_y[:,:,-2].unsqueeze(2), loan_done_mask)
                        reward_loss = loss_func(reward_prediction, b_y[:,:,-1].unsqueeze(2),reward_mask )

                        #multi-task loss controlled by the factor alpha
                        # loss = alpha * state_loss + (1-alpha) * reward_loss
                        loss = 0.25 * state_loss + 0.25 * installment_done_loss + 0.25 * loan_done_loss + 0.25 * reward_loss

                        optimizer.zero_grad()                   # clear gradients for this training step
                        loss.backward()                         # backpropagation, compute gradients
                        optimizer.step() 

                        train_loss_record.append(loss.item())

                        #if step % 500 ==0:
                        #    print (epoch, step, loss.item())
                            
                
                #torch.save(model, "generated_multitask_GRU_simulator_stratified.pkl")
                input_data = valid_data.values[:,1:input_size+1]
                s = valid_data.values[:,input_size+1: input_size+7]
                r = valid_data.values[:,-1]
                installment = valid_data.values[:,-3]
                loan = valid_data.values[:,-2]
                input_tensor = torch.tensor(input_data).float()
                input_tensor = Variable(torch.unsqueeze(input_tensor, 1))
                length = torch.ones(input_tensor.size(0), device = device).float()
                pred_s, pred_installment, pred_loan, pred_r, h_state = model(input_tensor,length,h_state)
                pred_s = pred_s.detach().cpu().numpy().squeeze()
                mse_s = mean_squared_error(pred_s, s)
                pred_r = pred_r.detach().cpu().numpy().squeeze()
                mse_r = mean_squared_error(pred_r, r)
                pred_installment = pred_installment.detach().cpu().numpy().squeeze()
                ce_installment = cross_entropy(installment, pred_installment)
                pred_loan = pred_loan.detach().cpu().numpy().squeeze()
                ce_loan = cross_entropy(loan, pred_loan)
                
                print('###############################################')
                print('parameters: learning rate = {}, batch size = {}, hidden size = {}'.format(LR, batch_size, hidden_size))
                print('loss on validation data ', 0.25 * mse_s + 0.25 * mse_r + 0.25 * ce_installment + 0.25 * ce_loan)