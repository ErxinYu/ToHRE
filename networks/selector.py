import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Selector(nn.Module):
	def __init__(self, config, relation_dim):
		super(Selector, self).__init__()
		self.config = config
		self.relation_matrix = nn.Embedding(self.config.flat_num_classes, relation_dim) 
		self.bias = nn.Parameter(torch.Tensor(self.config.flat_num_classes))
		self.attention_matrix = nn.Embedding(self.config.global_num_classes, relation_dim) #[95,690]
		self.attention_matrix_flat = nn.Embedding(self.config.flat_num_classes, relation_dim) #[95,690]
		self.init_weights()
		self.scope = None
		self.attention_query = None
		self.attention_query_flat = None
		self.test_attention_query = None
		self.label = None
		self.dropout = nn.Dropout(self.config.base_model_drop_prob)
		self.layer_tensor = torch.arange(3).long().cuda()



	def init_weights(self):	
		nn.init.xavier_uniform_(self.relation_matrix.weight.data)
		nn.init.normal_(self.bias)
		nn.init.xavier_uniform_(self.attention_matrix.weight.data)
		nn.init.xavier_uniform_(self.attention_matrix_flat.weight.data)
	def get_logits(self, x):
		logits = torch.matmul(x, torch.transpose(self.relation_matrix.weight, 0, 1),) + self.bias # [53,53] = [53, 2070] * [2070, 53]
		return logits
	def forward(self, x):
		raise NotImplementedError
	def test(self, x):
		raise NotImplementedError

class Attention(Selector):
	def _attention_train_logit(self, x, layer):
		attention = self.attention_matrix(self.attention_query[:,layer]) #[all_sen_num, 690]
		attention_logit = torch.sum(x * attention, 1, True)# x = [all_sen_num, 690]
		return attention_logit

	def _attention_train_logit_flat(self, x):
		relation_query = self.relation_matrix(self.attention_query_flat)
		attention = self.attention_matrix(self.attention_query_flat)
		#print("")
		#print("x * attention", (x * attention).size())
		#print("x * attention * relation_query", (x * attention * relation_query).size())
		attention_logit_flat = torch.sum(x * attention * relation_query, 1, True)
		#attention_logit = torch.sum(x * attention, 1, True)
		return attention_logit_flat

	def _attention_test_logit(self, x):
		attention_logit = torch.matmul(x, torch.transpose(self.attention_matrix.weight, 0, 1)) #[all_sen_num, 95] = [all_sen_num, 690] * [690, 95]
		return attention_logit

	def _attention_test_logit_flat(self, x):
		attention_logit_flat = torch.matmul(x, torch.transpose(self.attention_matrix_flat.weight * self.relation_matrix.weight, 0, 1))
		#attention_logit = torch.matmul(x, torch.transpose(self.attention_matrix.weight, 0, 1))

		return attention_logit_flat

	def forward(self, x): #[all_sen_num, 690]
		logits_layers = []
		logits_total = None
		probs =None
		for layer in range(3):
			attention_logit = self._attention_train_logit(x, layer) #[all_sen_num, 1]
			tower_repre = []
			for i in range(len(self.scope) - 1):
				sen_matrix = x[self.scope[i] : self.scope[i + 1]]
				attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i] : self.scope[i + 1]], 0, 1), 1) #[1, bag_sen_num]
				final_repre = torch.squeeze(torch.matmul(attention_score, sen_matrix))# [1, bag_sen_num] * [bag_sen_num, 690]
				tower_repre.append(final_repre) #[1, 690]
			stack_repre_layer = torch.stack(tower_repre) #[batch, 690]
			stack_repre_layer = self.dropout(stack_repre_layer)
			logits_layers.append(stack_repre_layer)  #[laye_0‘s (batch, 690), laye_1‘s (batch, 690),laye_2‘s (batch, 690)]
		stack_logits_layers = torch.stack(logits_layers) #list 变tensor.  tensor( laye_0‘s (batch, 690), laye_1‘s (batch, 690),laye_2‘s (batch, 690))
		# if self.config.global_ratio > 0:
		# 	logits_total = torch.cat(logits_layers, 1)
		# 	logits_total = self.dropout(logits_total)
		# 	probs = self.get_logits(logits_total)
		return stack_logits_layers, logits_total, probs

	def forward_flat(self, x):
		attention_logit_flat = self._attention_train_logit_flat(x)
		tower_repre = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]]
			attention_score = F.softmax(torch.transpose(attention_logit_flat[self.scope[i] : self.scope[i + 1]], 0, 1), 1)
			final_repre = torch.squeeze(torch.matmul(attention_score, sen_matrix))
			tower_repre.append(final_repre)
		stack_repre = torch.stack(tower_repre)
		stack_repre = self.dropout(stack_repre)
		logits = self.get_logits(stack_repre)
		return logits

	def test_hierarchical(self, x): #x = [all_sen_num,690]
		attention_logit = self._attention_test_logit(x)	#[all_sen_num, 95]	
		tower_repre = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]] #[bag_sen_num, 690]
			attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i] : self.scope[i + 1]], 0, 1), 1) #[95,bag_sen_num]
			logits = torch.matmul(attention_score, sen_matrix) #[95, 690] = [95,bag_sen_num] * [bag_sen_num, 690]
			tower_repre.append(logits)
		stack_repre = torch.stack(tower_repre)#[160,95,690]
		return stack_repre

	def test_flat(self, x):
		attention_logit_flat = self._attention_test_logit_flat(x)	
		tower_output = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]]
			attention_score = F.softmax(torch.transpose(attention_logit_flat[self.scope[i] : self.scope[i + 1]], 0, 1), 1)
			final_repre = torch.matmul(attention_score, sen_matrix)
			logits = self.get_logits(final_repre)
			tower_output.append(torch.diag(F.softmax(logits, 1)))
		stack_output = torch.stack(tower_output)
		return list(stack_output.data.cpu().numpy())
 
	# def test_flat(self, x):
	# 	attention_logit = self._attention_test_logit(x)	#[all_sen_num, 95]
	# 	tower_output = []
	# 	for i in range(len(self.scope) - 1):
	# 		sen_matrix = x[self.scope[i] : self.scope[i + 1]]
	# 		attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i] : self.scope[i + 1]], 0, 1), 1) #[96,bag_sen_num]
	# 		final_repre = torch.matmul(attention_score, sen_matrix) #[95, 690] = [95,bag_sen_num] * [bag_sen_num, 690]
	# 		total_label_repre = []
	# 		for i in range(53):
	# 			each_label_repre = torch.index_select(final_repre, 0, self.test_attention_query[i]).view(1,-1) #[3,690]--> [1,2070]
	# 			total_label_repre.append(each_label_repre)
	# 			#print("each_label_repre",each_label_repre.size())
	# 			#print("total_label_repre",len(total_label_repre), total_label_repre[0].size())
	# 		stack_repre = torch.stack(total_label_repre) #[53, 1, 2070]
	# 		stack_repre = torch.squeeze(stack_repre)
	# 		#print("stack_repre", stack_repre.size())
	# 		probs = self.get_logits(stack_repre) #[53,53] = [53,2070] * [2070,53]
	# 		#print("probs", probs, probs.size())
	# 		tower_output.append(torch.diag(F.softmax(probs, 1))) #[1,53]
	# 	stack_output = torch.stack(tower_output) #[batch,53]
	# 	return list(stack_output.data.cpu().numpy())