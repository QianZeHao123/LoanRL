import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence 
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('Data/20240205fullsample_new.csv')

data['installment_timestep'] = data.groupby(['loan_id','installment']).cumcount()+1

states=['installment','installment_timestep','state_cum_overduelength','remaining_debt','state_capital','state_interests',
        'state_penalty','gender','age','amount','num_loan','duration','year_ratio','diff_city',
        'marriage','kids','month_in','housing','edu','motivation']
#print('# of states ', len(states))

data[states] = (data[states] - data[states].mean()) / data[states].std()
#print(data.head())
#pdb.set_trace()

train = data.loc[(data['sample'] == 'rlsimulator') & (data['group'] == 'train')]
test = data.loc[(data['sample'] == 'rlsimulator') & (data['group'] == 'test')]

data_list = [train, test]
data_name_list = ['train', 'test']

varstate_size = 7

for j in tqdm(range(len(data_list)), leave=True):
    dt = data_list[j]
    dt_loan_ids = dt['loan_id'].drop_duplicates().tolist()

    X_df = pd.DataFrame()
    y_df = pd.DataFrame()

    for loan_id in tqdm(dt_loan_ids, leave=True):
        df1 = dt.loc[dt['loan_id'] == loan_id]
        
        X_train = df1[['loan_id'] + states + ['action_num_actual','installment_done','loan_done','recovery_rate_weighted']]
        # X_train = X_train[:-1]
        X_df = X_df.append(X_train, ignore_index = True)

        y_train = df1[states[:varstate_size]]
        y_train = y_train.rename(columns = {'installment':'installment.1', 'installment_timestep': 'installment_timestep.1', 'state_cum_overduelength': 'state_cum_overduelength.1', 'remaining_debt': 'remaining_debt.1', 'state_capital': 'state_capital.1','state_interests': 'state_interests.1', 'state_penalty':'state_penalty.1'})
            
        if y_train.shape[0] > 1:
            y_train = y_train[1:]
            y_train = y_train.append(y_train.iloc[-1], ignore_index = True)
            
        y_df = y_df.append(y_train, ignore_index = True)

    cols = ['loan_id'] + states + ['action_num_actual'] + ['installment.1','installment_timestep.1','state_cum_overduelength.1','remaining_debt.1', 'state_capital.1', 'state_interests.1','state_penalty.1'] + ['installment_done','loan_done','recovery_rate_weighted']
    result = pd.concat([X_df, y_df], axis = 1)[cols]
    #print('result shape ', result.shape)

    ids = np.unique(result["loan_id"])
    for i in tqdm(range(len(ids)), leave=True): 
        result.loc[result['loan_id'] == ids[i], 'loan_id']= i
        
    result['loan_id'] = [int(x) for x in result['loan_id']]
        
    result.to_csv('Simulator/simulator_'+data_name_list[j]+'_data.csv', index = False)

    ##normalization
    # Compile the loans into batches
    batch_data = {}
    ids = np.unique(result["loan_id"])
    normalized = result.values

    for i in tqdm(range(len(ids)), leave=True):
        id_i = ids[i]
        batch_data[i] = normalized[normalized[:,0]==id_i]


    output = open('Simulator/simulator_'+data_name_list[j]+'_batch.pkl', 'wb')
    pickle.dump(batch_data, output)
    output.close()
    #pdb.set_trace()

ids = result['loan_id'].drop_duplicates().tolist()
k = int(0.1 * len(ids)) # 10% validation data

random.seed(42) # the current train and valid data are sampled without setting the random seed. if we run this later, we may get different dataset. 
valid_ids = random.sample(ids, k)
train_ids = [elem for elem in ids if elem not in valid_ids]

train_data = result.loc[result['loan_id'].isin(train_ids)]
valid_data = result.loc[result['loan_id'].isin(valid_ids)]
valid_data.to_csv('Simulator/simulator_training_valid_batch.csv', index = False)
#pdb.set_trace()

batch_data = {}
normalized = train_data.values

for i in tqdm(range(len(train_ids)), leave=True):
    id_i = train_ids[i]
    batch_data[i] = normalized[normalized[:,0]==id_i]


output = open('Simulator/simulator_training_train_batch.pkl', 'wb')
pickle.dump(batch_data, output)
output.close()

#batch_data = {}
#normalized = valid_data.values

#for i in range(len(valid_ids)):
#    id_i = valid_ids[i]
#    batch_data[i] = normalized[normalized[:,0]==id_i]


#output = open('simulator_training_valid_batch.pkl', 'wb')
#pickle.dump(batch_data, output)
#output.close()



