import random
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import math
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence 
from argparse import ArgumentParser
from collections import Counter
from quantization_uncertainty_exploration import *
# from doubledueling_dqn import *
import pdb
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, auc


def calculate_metrics(predicted, actual):
    TP = np.sum((predicted == 1) & (actual == 1))
    FP = np.sum((predicted == 1) & (actual == 0))
    FN = np.sum((predicted == 0) & (actual == 1))

    accuracy = np.mean(predicted == actual)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score

def plot_roc_curve(actual_labels_train, predicted_probs_train, actual_labels_test, predicted_probs_test, title_label, save_filename = None):
    fpr_train, tpr_train, _ = roc_curve(actual_labels_train, predicted_probs_train)
    roc_auc_train = auc(fpr_train, tpr_train)
    
    fpr_test, tpr_test, _ = roc_curve(actual_labels_test, predicted_probs_test)
    roc_auc_test = auc(fpr_test, tpr_test)

    plt.figure()
    plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label='Train ROC curve (area = %0.2f)' % roc_auc_train)
    plt.plot(fpr_test, tpr_test, color='blue', lw=2, label='Test ROC curve (area = %0.2f)' % roc_auc_test)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC Curve'.format(title_label))
    plt.legend(loc="lower right")
    
    if save_filename:
        plt.savefig(save_filename)
        print(f"ROC curve plot saved as {save_filename}")
    else:
        plt.show()

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_states_action = 21
varstate_size = 7

# Load training data
train_data = pd.read_csv('Simulator/simulator_train_data.csv')
train_input_data = train_data.values[:,1:n_states_action+1]
train_s = train_data.values[:,n_states_action+1: n_states_action+1+varstate_size]
train_installment = train_data.values[:,-3]
train_loan = train_data.values[:,-2]
train_r = train_data.values[:,-1]

# Load testing data
test_data = pd.read_csv('Simulator/simulator_test_data.csv')
test_input_data = test_data.values[:,1:n_states_action+1]
test_s = test_data.values[:,n_states_action+1: n_states_action+1+varstate_size]
test_installment = test_data.values[:,-3]
test_loan = test_data.values[:,-2]
test_r = test_data.values[:,-1]

for model in ['LSTM', 'RNN','GRU']:
    print('************************** Start {} *****************************'.format(model))

    sim_LR = 0.001
    sim_batch_size = 64
    simulator = torch.load('Simulator/generated_multitask_'+model+'_simulator_'+str(sim_LR)+'_'+str(sim_batch_size)+'.pkl')
    #simulator = torch.load('Simulator/generated_multitask_'+model+'_simulator.pkl')
    simulator.to(device)
    
    # Predictions for training data
    train_input_tensor = torch.tensor(train_input_data).float()
    train_input_tensor = Variable(torch.unsqueeze(train_input_tensor, 1))
    train_length = torch.ones(train_input_tensor.size(0), device = device).float()
    h_state = None

    simulator.eval()
    with torch.no_grad(): 
        train_pred_s, train_pred_installment, train_pred_loan, train_pred_r, h_state = simulator(train_input_tensor, train_length, h_state)
        train_pred_s = train_pred_s.detach().cpu().numpy().squeeze()
        mse_s = mean_squared_error(train_pred_s, train_s)
        train_pred_r = train_pred_r.detach().cpu().numpy().squeeze()
        mse_r = mean_squared_error(train_pred_r, train_r)
        train_pred_installment = train_pred_installment.detach().cpu().numpy().squeeze()
        auc_installment_train = roc_auc_score(train_installment, train_pred_installment)
        train_predicted = np.where(train_pred_installment > 0.5, 1, 0)
        acc_installment_train, precision_installment_train, recall_installment_train, f1_installment_train = calculate_metrics(train_predicted, train_installment)
        train_pred_loan = train_pred_loan.detach().cpu().numpy().squeeze()
        auc_loan_train = roc_auc_score(train_loan, train_pred_loan)
        train_predicted = np.where(train_pred_loan > 0.5, 1, 0)
        acc_loan_train, precision_loan_train, recall_loan_train, f1_loan_train = calculate_metrics(train_predicted, train_loan)

    print('Training state prediction mse ', mse_s)
    print('Training reward prediction mse ', mse_r)
    print('##################################################################')
    print('Training installment prediction auc ', auc_installment_train)
    print('Training installment prediction accuracy ', acc_installment_train)
    print('Training installment prediction precision ', precision_installment_train)
    print('Training installment prediction recall ', recall_installment_train)
    print('Training installment prediction f1 score ', f1_installment_train)
    print('##################################################################')
    print('Training loan prediction auc ', auc_loan_train)
    print('Training loan prediction accuracy ', acc_loan_train)
    print('Training loan prediction precision ', precision_loan_train)
    print('Training loan prediction recall ', recall_loan_train)
    print('Training loan prediction f1 score ', f1_loan_train)

    # Predictions for testing data
    test_input_tensor = torch.tensor(test_input_data).float()
    test_input_tensor = Variable(torch.unsqueeze(test_input_tensor, 1))
    test_length = torch.ones(test_input_tensor.size(0), device = device).float()
    h_state = None

    with torch.no_grad(): 
        test_pred_s, test_pred_installment, test_pred_loan, test_pred_r, h_state = simulator(test_input_tensor, test_length, h_state)
        test_pred_s = test_pred_s.detach().cpu().numpy().squeeze()
        mse_s_test = mean_squared_error(test_pred_s, test_s)
        test_pred_r = test_pred_r.detach().cpu().numpy().squeeze()
        mse_r_test = mean_squared_error(test_pred_r, test_r)
        test_pred_installment = test_pred_installment.detach().cpu().numpy().squeeze()
        auc_installment_test = roc_auc_score(test_installment, test_pred_installment)
        test_predicted = np.where(test_pred_installment > 0.5, 1, 0)
        acc_installment_test, precision_installment_test, recall_installment_test, f1_installment_test = calculate_metrics(test_predicted, test_installment)
        test_pred_loan = test_pred_loan.detach().cpu().numpy().squeeze()
        auc_loan_test = roc_auc_score(test_loan, test_pred_loan)
        test_predicted = np.where(test_pred_loan > 0.5, 1, 0)
        acc_loan_test, precision_loan_test, recall_loan_test, f1_loan_test = calculate_metrics(test_predicted, test_loan)

    print('Testing state prediction mse ', mse_s_test)
    print('Testing reward prediction mse ', mse_r_test)
    print('##################################################################')
    print('Testing installment prediction auc ', auc_installment_test)
    print('Testing installment prediction accuracy ', acc_installment_test)
    print('Testing installment prediction precision ', precision_installment_test)
    print('Testing installment prediction recall ', recall_installment_test)
    print('Testing installment prediction f1 score ', f1_installment_test)
    print('##################################################################')
    print('Testing loan prediction auc ', auc_loan_test)
    print('Testing loan prediction accuracy ', acc_loan_test)
    print('Testing loan prediction precision ', precision_loan_test)
    print('Testing loan prediction recall ', recall_loan_test)
    print('Testing loan prediction f1 score ', f1_loan_test)

    plot_roc_curve(train_installment, train_pred_installment, test_installment, test_pred_installment, 'Installment', save_filename='Simulator/Figure/CV' + model+'_ROC_installment.png')
    plot_roc_curve(train_loan, train_pred_loan, test_loan, test_pred_loan, 'Loan', save_filename='Simulator/Figure/CV' + model+'_ROC_loan.png')
