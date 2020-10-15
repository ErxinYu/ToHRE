import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PCNN(nn.Module):
	def __init__(self, config): #
		super(PCNN, self).__init__()
		self.config = config
		self.mask = None
		self.cnn = _CNN(config)
		self.pooling = _PiecewisePooling()
		self.activation = nn.ReLU()
		self.attention_query = None
		self.bag_ids = None
		self.dropout = nn.Dropout(self.config.base_model_drop_prob)
	def forward(self, embedding): # embedding [241,120,60]
		embedding = torch.unsqueeze(embedding, dim = 1) #[241,1,120,60]
		x = self.cnn(embedding) #[241,300,120,1]
		# print("encoder cnn x", x.size())
		x = self.pooling(x, self.mask, self.config.hidden_size) #[241, 690]
		#print("encoder pooling x", x.size())
		x = self.activation(x)
		x = self.dropout(x)
		#exit()
		return x
class _CNN(nn.Module):
	def __init__(self, config):
		super(_CNN, self).__init__()
		self.config = config
		self.in_channels = 1
		self.in_height = self.config.max_length #120
		self.in_width = self.config.word_size + 2 * self.config.pos_size #60
		self.kernel_size = (self.config.window_size, 105) #（3, 60/105）
		self.out_channels = self.config.hidden_size
		self.stride = (1, 1)
		self.padding = (1, 0)
		self.cnn = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
	def forward(self, embedding):
		return self.cnn(embedding)


class _PiecewisePooling(nn.Module):
	def __init(self):
		super(_PiecewisePooling, self).__init__()
	def forward(self, x, mask, hidden_size): #x [244, 300, 120, 1]  mask [244,120,3]
		mask = torch.unsqueeze(mask, 1) # [244,1,120,3]
		# print("x", x[0])
		# print("mask", mask[0])
		# print("mask+x", (mask+x)[0], (mask+x).size())
		x, _ = torch.max(mask + x, dim = 2) # mask+x = [244,300,120,3].      x = [224,300,3]
		# print("x", x[0], x.size())
		x = x - 100
		return x.view(-1, hidden_size * 3) #[224,900]