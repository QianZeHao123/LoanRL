# The full descriptions of the key hyper-parameters are as follows:
# - input_size: the input size of the simulator
# - lr : The learning rate
# - batch_size : The size of the batch when sample from memory experience
# - hidden_size : the hidden size of the base deep learning model, e.g., LSTM/GRU/RNN
# - alpha: the coefficient of the state prediction task for multitask training
# - epoch: number of training epochs
# - training_data_path: training data absolute path
#      - which_model: select the base model, options are 'LSTM', 'GRU', 'RNN'

python multitask_simulator_training.py --input_size 20 \
	--lr 0.001 \
	--batch_size 16 \
	--hidden_size 50 \
	--alpha 0.25 \
	--epoch 100 \
	--training_data_path simulator_training_batch.pkl \
	--which_model 'LSTM'

python multitask_simulator_training.py --input_size 20 \
	--lr 0.001 \
	--batch_size 64 \
	--hidden_size 50 \
	--alpha 0.25 \
	--epoch 100 \
	--training_data_path simulator_training_batch.pkl \
	--which_model 'RNN'

python multitask_simulator_training.py --input_size 20 \
	--lr 0.001 \
	--batch_size 64 \
	--hidden_size 50 \
	--alpha 0.25 \
	--epoch 100 \
	--training_data_path simulator_training_batch.pkl \
	--which_model 'GRU'
	
