import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from networks.embedding import *
from networks.encoder import *
from networks.selector import *
from networks.classifier import *

class PCNN_ATT(nn.Module):
	def __init__(self, config):	
		super(PCNN_ATT, self).__init__()
		self.config = config
		self.embedding = Embedding(config)
		self.encoder = PCNN(config)
		self.selector = Attention(config, config.hidden_size * 3)

	def forward(self):
		embedding = self.embedding()
		sen_embedding = self.encoder(embedding)
		logits_layers, logits_total, flat_probs = self.selector(sen_embedding)
		return logits_layers, logits_total, flat_probs

	def forward_flat(self):
		embedding = self.embedding()
		sen_embedding = self.encoder(embedding)
		logits = self.selector.forward_flat(sen_embedding)
		return logits

	def test_flat(self):
		embedding = self.embedding()
		sen_embedding = self.encoder(embedding)
		return self.selector.test_flat(sen_embedding)

	def test_hierarchical(self):
		embedding = self.embedding()
		sen_embedding = self.encoder(embedding)
		return self.selector.test_hierarchical(sen_embedding)