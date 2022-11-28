import torch 

num_epochs = 10
batch_size = 128
learning_rate = 0.001
beta1 = 0.9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
configurations = {
	'num_epochs': num_epochs,
	'batch_size': batch_size,
	'learning_rate': learning_rate,
	'beta1': beta1,
	'device': device
}