import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Embedding(nn.Module):
	def __init__(self, config):
		super(Embedding, self).__init__()
		self.config = config
		self.word_embedding = nn.Embedding(self.config.data_word_vec.shape[0], self.config.data_word_vec.shape[1])
		self.pos1_embedding = nn.Embedding(self.config.pos_num, self.config.pos_size, padding_idx = 0)
		self.pos2_embedding = nn.Embedding(self.config.pos_num, self.config.pos_size, padding_idx = 0)
		self.init_word_weights()
		self.init_pos_weights()
		self.word = None
		self.pos1 = None
		self.pos2 = None


		self.h_entity_word = None
		self.t_entity_word = None
		self.linear = nn.Linear(105, 105)

		# self.linear_t = nn.Linear(105,150)
		# self.linear_a = nn.Linear(150,150)

		self.activation = nn.Sigmoid()

	def init_word_weights(self):
		self.word_embedding.weight.data.copy_(torch.from_numpy(self.config.data_word_vec))
	
	def init_pos_weights(self):
		nn.init.xavier_uniform_(self.pos1_embedding.weight.data)
		if self.pos1_embedding.padding_idx is not None:
			self.pos1_embedding.weight.data[self.pos1_embedding.padding_idx].fill_(0)
		nn.init.xavier_uniform_(self.pos2_embedding.weight.data)
		if self.pos2_embedding.padding_idx is not None:
			self.pos2_embedding.weight.data[self.pos2_embedding.padding_idx].fill_(0)
	def forward_old(self):
		word = self.word_embedding(self.word)	#[160,120,50]
		pos1 = self.pos1_embedding(self.pos1) #[160,120,5]
		pos2 = self.pos2_embedding(self.pos2) #[160,120,5]
		embedding = torch.cat((word, pos1, pos2), dim = 2) #[160,120,60]
		return embedding
	def forward_v1(self):

		# print("word", self.word.size(), self.word[100])
		# print("pos1", self.pos1.size(), self.pos1[100])
		# print("pos2", self.pos2.size(), self.pos2[100])
		# print("h_entity", self.h_entity_word.size(), self.h_entity_word[100])
		# print("t_entity", self.t_entity_word.size(), self.t_entity_word[100])
		

		word = self.word_embedding(self.word)	#[160,120,50]
		h_entity_word = self.word_embedding(self.h_entity_word)	#[160,1,50]
		t_entity_word = self.word_embedding(self.t_entity_word) #[160,1,50]
		h_entity_word = h_entity_word.expand(-1,120,50) #[160,120,50]
		t_entity_word = t_entity_word.expand(-1,120,50) #[160,120,50]
		pos1 = self.pos1_embedding(self.pos1) #[160,120,5]
		pos2 = self.pos2_embedding(self.pos2) #[160,120,5]

		# print("word", word.size())
		# print("pos1", pos1.size())
		# print("pos2", pos2.size(), self.pos2[100])
		# print("h_entity", h_entity_word.size(), h_entity_word[100])
		#print("t_entity", t_entity_word.size(), t_entity_word[100])


		embedding_h = torch.cat((word, h_entity_word, pos1, pos2), dim = 2) #[160,120,110]
		embedding_t = torch.cat((word, t_entity_word, pos1, pos2), dim = 2) #[160,120,110]
		# print("embedding_h", embedding_h.size(), embedding_h[100])
		# print("embedding_t", embedding_t.size(), embedding_t[100])
		a1 = self.linear(embedding_h) #[160,120,110]
		a1 = self.activation(a1)
		a2 = 1 - a1 #[160,120,110]
		embedding_h = embedding_h.mul(a1) #[160,120,110]
		embedding_t = embedding_t.mul(a2) #[160,120,110]
		embedding = embedding_h + embedding_t #[160,120,110]
		return embedding
	def forward(self):
		
		word = self.word_embedding(self.word)	#[160,120,50]
		h_entity_word = self.word_embedding(self.h_entity_word)	#[160,1,50]
		t_entity_word = self.word_embedding(self.t_entity_word) #[160,1,50]
		h_entity_word = h_entity_word.expand(-1,120,50) #[160,120,50]
		t_entity_word = t_entity_word.expand(-1,120,50) #[160,120,50]
		pos1 = self.pos1_embedding(self.pos1) #[160,120,5]
		pos2 = self.pos2_embedding(self.pos2) #[160,120,5]

		embedding_h = torch.cat((word, h_entity_word, pos1), dim = 2) #[160,120,105]
		embedding_t = torch.cat((word, t_entity_word, pos2), dim = 2) #[160,120,105]

		a1 = self.linear(embedding_h) #[160,120,105]
		a1 = self.activation(a1)
		a2 = 1 - a1 #[160,120,105]
		embedding_h = embedding_h.mul(a1) #[160,120,105]
		embedding_t = embedding_t.mul(a2) #[160,120,105]
		embedding = embedding_h + embedding_t #[160,120,105]
		return embedding
		
	def forward_v3(self):



		word = self.word_embedding(self.word)	#[160,120,50]
		h_entity_word = self.word_embedding(self.h_entity_word)	#[160,1,50]
		t_entity_word = self.word_embedding(self.t_entity_word) #[160,1,50]
		h_entity_word = h_entity_word.expand(-1,120,50) #[160,120,50]
		t_entity_word = t_entity_word.expand(-1,120,50) #[160,120,50]
		pos1 = self.pos1_embedding(self.pos1) #[160,120,5]
		pos2 = self.pos2_embedding(self.pos2) #[160,120,5]



		embedding_h = torch.cat((word, h_entity_word, pos1), dim = 2) #[160,120,105]
		embedding_t = torch.cat((word, t_entity_word, pos2), dim = 2) #[160,120,105]

		embedding_h = self.linear_t(embedding_h) #[160,120,200]
		embedding_t = self.linear_t(embedding_t)



		a1 = self.linear_a(embedding_h) #[160,120,200]
		a1 = self.activation(a1)
		a2 = 1 - a1 #[160,120,200]
		embedding_h = embedding_h.mul(a1) #[160,120,200]
		embedding_t = embedding_t.mul(a2) #[160,120,200]
		embedding = embedding_h + embedding_t #[160,120,200]

		return embedding	
