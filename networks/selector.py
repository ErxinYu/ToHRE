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
		self.relation_matrix = nn.Embedding(self.config.flat_num_classes, relation_dim *3) 
		self.bias = nn.Parameter(torch.Tensor(self.config.flat_num_classes))
		self.attention_matrix = nn.Embedding(self.config.global_num_classes, relation_dim) #[95,690]
		self.init_weights()
		self.scope = None
		self.attention_query = None
		self.test_attention_query = None
		self.label = None
		self.dropout = nn.Dropout(self.config.drop_prob)
		self.layer_tensor = torch.arange(3).long().cuda()
		#print("self.layer_tensor", self.layer_tensor)
	def init_weights(self):	
		nn.init.xavier_uniform_(self.relation_matrix.weight.data)
		nn.init.normal_(self.bias)
		nn.init.xavier_uniform_(self.attention_matrix.weight.data)
	def get_logits(self, x):
		logits = torch.matmul(x, torch.transpose(self.relation_matrix.weight, 0, 1),) + self.bias # [53,53] = [53, 2070] * [2070, 53]
		return logits
	def forward(self, x):
		raise NotImplementedError
	def test(self, x):
		raise NotImplementedError

class Attention_word_att(Selector): 
	def _attention_train_logit(self, x, layer): #x [all_sen_num, 690]
		#print("self.attention_query[:,layer]", self.attention_query[:,layer].size(), self.attention_query[:,layer].type())
		attention = self.attention_matrix(self.attention_query[:,layer]) #[all_sen_num, 690]
		#print("attention", attention.size(), attention.type())

		attention_logit = torch.sum(x * attention, 1, True)# [all_sen_num, 690]=[all_sen_num, 690]*[all_sen_num, 690] sum:[all_sen_num,1]
		#print("attention_logit", attention_logit.size())
		return attention_logit
	def _attention_test_logit(self, x):
		attention_logit = torch.matmul(x, torch.transpose(self.attention_matrix.weight, 0, 1)) #[all_sen, 95, 95] = [all_sen_num, 95,690] * [690, 95]
		return attention_logit
	def forward(self, x):# x[all_sen_num, 3, 690]
		#print("x_all", x.size())   #[all_sen_num, 3, 690]
		logits_layers = []
		for layer in range(3):
			#print("\n")
			#print("x",  x.size()) 
			#先进行关系的筛选。输入x[all_sen_num, 3 690], self.attention_query[all_sen,1] 输出x_rel_spe [all_sen, 690]
			#print("self.attention_query", self.attention_query.size())
			#print("self.layer_tensor", self.layer_tensor)
			indices = self.layer_tensor[layer].view(1).long() #[1]
			#print("indices_before", indices, indices.size())
			indices = indices.unsqueeze(0) #[1,1]
			#print("indices_1", indices, indices.size())
			indices = indices.unsqueeze(2).expand(x.size(0), 1, x.size(2)) #[all_sen,1,690]
			#print("indices_after", indices, indices.size())
			#print("indices", indices, indices.size(), indices.type())   #[all_sen, 1,690]
			x_rel_spe = torch.gather(x, dim=1, index=indices) #x_rel_spe=[all_sen,1,690],  x=[all_sen,3,690] indices=[all_sen,1,690]  
			
			x_rel_spe = x_rel_spe.squeeze() #[all_sen, 690]
			#print("x_rel_spe", x_rel_spe.size())
			attention_logit = self._attention_train_logit(x_rel_spe,layer) #[all_sen_num, 1]
			#print("attention_logit", attention_logit.size())
			layer_repre = []
			for i in range(len(self.scope) - 1):
				#print("layer", layer)
				#print("indices", indices.size())
				sen_matrix = x_rel_spe[self.scope[i] : self.scope[i + 1]] #[bag_sen_num, 690]
				#print("sen_matrix", sen_matrix.size())
				attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i] : self.scope[i + 1]], 0, 1), 1) #[1, bag_sen_num]
				#print("attention_score", attention_score.size())
				final_repre = torch.matmul(attention_score, sen_matrix) # [1,690] = [1, bag_sen_num] * [bag_sen_num, 690]
				final_repre = torch.squeeze(final_repre)# [690]
				#print("final_repre",final_repre.size())
				layer_repre.append(final_repre) 

			stack_layer_layer = torch.stack(layer_repre) #[batch, 690]
			#print("stack_repre_layer", stack_repre_layer.size())
			logits_layers.append(stack_layer_layer)  #[laye_0‘s [batch, 690], laye_1‘s [batch, 690],laye_2‘s [batch, 690]]
		#stack_logits_layers = torch.stack(logits_layers)
		#print("stack_logits_layers", stack_logits_layers.size(), type(stack_logits_layers))
		#print("logits_layers", len(logits_layers), type(logits_layers))
		#exit()
		logits_total = torch.cat(logits_layers, 1) #[batch, 690*3]
		logits_total = self.dropout(logits_total)
		probs = self.get_logits(logits_total) #[batch, 53]
		return logits_layers, logits_total, probs

	def test_hierarchical(self, x): #[all_sen_num, 95,690]
		#print("----x---", x.size())
		attention_logit = self._attention_test_logit(x)	#[all_sen_num, 95, 95]
		#print("attention_logit_before", attention_logit[0], attention_logit.size())
		attention_logit = torch.diagonal(attention_logit, offset=0, dim1=-2, dim2=-1)#[all_sen_num, 95]
		#print("attention_logit_after", attention_logit[0], attention_logit.size())
		#exit()
		tower_repre = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]] #[bag_sen_num, 95,690]
			#print("sen_matrix_before", sen_matrix, sen_matrix.size())

			sen_matrix = sen_matrix.permute(1,0,2)  #[95, bag_sen_num, 690]
			#print("sen_matrix_after", sen_matrix.size())

			attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i] : self.scope[i + 1]], 0, 1), 1) #[95,bag_sen_num]
			#print("attention_score", attention_score[1], attention_score.size())

			logits = torch.matmul(attention_score, sen_matrix) #[95,95, 690] = [95, bag_sen_num] * [95, bag_sen_num,690]
			#print("logits_before", logits[1][1], logits.size())
			logits = torch.diagonal(logits, offset=0, dim1=0, dim2=1)#[95, 690]
			#print("logits_after", logits.size())
			logits = torch.transpose(logits, 0, 1)
			#print("logits_final", logits[1],logits.size())
			#exit()
			tower_repre.append(logits)
		stack_repre = torch.stack(tower_repre)#[160,95,690]
		#print("stack_repre", stack_repre.size())
		return stack_repre


	def test_hierarchical_ori(self, x): #x = [all_sen_num,690]
		attention_logit = self._attention_test_logit(x)	#[all_sen_num, 95]
		tower_repre = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]] #[bag_sen_num, 690]
			attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i] : self.scope[i + 1]], 0, 1), 1) #[95,bag_sen_num]
			logits = torch.matmul(attention_score, sen_matrix) #[95, 690] = [95,bag_sen_num] * [bag_sen_num, 690]
			tower_repre.append(logits)
		stack_repre = torch.stack(tower_repre)#[160,95,690]
		return stack_repre











class Attention(Selector):
	def _attention_train_logit(self, x, layer):
		attention = self.attention_matrix(self.attention_query[:,layer]) #[all_sen_num, 690]
		attention_logit = torch.sum(x * attention, 1, True)# x = [all_sen_num, 690]
		return attention_logit
	def _attention_test_logit(self, x):
		attention_logit = torch.matmul(x, torch.transpose(self.attention_matrix.weight, 0, 1)) #[all_sen_num, 95] = [all_sen_num, 690] * [690, 95]
		return attention_logit
	def forward(self, x): #[all_sen_num, 690]
		#print("selector x",x.size())
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
			logits_layers.append(stack_repre_layer)  #[laye_0‘s (batch, 690), laye_1‘s (batch, 690),laye_2‘s (batch, 690)]
		stack_logits_layers = torch.stack(logits_layers) #list 变tensor.  tensor( laye_0‘s (batch, 690), laye_1‘s (batch, 690),laye_2‘s (batch, 690))
		if self.config.global_ratio > 0:
			logits_total = torch.cat(logits_layers, 1)
			logits_total = self.dropout(logits_total)
			probs = self.get_logits(logits_total)
		return stack_logits_layers, logits_total, probs

	def test_flat(self, x):
		attention_logit = self._attention_test_logit(x)	#[all_sen_num, 95]
		tower_output = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]]
			attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i] : self.scope[i + 1]], 0, 1), 1) #[96,bag_sen_num]
			final_repre = torch.matmul(attention_score, sen_matrix) #[95, 690] = [95,bag_sen_num] * [bag_sen_num, 690]
			total_label_repre = []
			for i in range(53):
				each_label_repre = torch.index_select(final_repre, 0, self.test_attention_query[i]).view(1,-1) #[3,690]--> [1,2070]
				total_label_repre.append(each_label_repre)
				#print("each_label_repre",each_label_repre.size())
				#print("total_label_repre",len(total_label_repre), total_label_repre[0].size())
			stack_repre = torch.stack(total_label_repre) #[53, 1, 2070]
			stack_repre = torch.squeeze(stack_repre)
			#print("stack_repre", stack_repre.size())
			probs = self.get_logits(stack_repre) #[53,53] = [53,2070] * [2070,53]
			#print("probs", probs, probs.size())
			tower_output.append(torch.diag(F.softmax(probs, 1))) #[1,53]
		stack_output = torch.stack(tower_output) #[batch,53]
		return list(stack_output.data.cpu().numpy())

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

class One(Selector):
	def forward(self, x):
		tower_logits = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]]
			sen_matrix = self.dropout(sen_matrix)
			print(sen_matrix,sen_matrix.size())
			logits = self.get_logits(sen_matrix)
			score = F.softmax(logits, 1)
			_, k = torch.max(score, dim = 0)
			print("logits",logits, len(logits))
			print("score", score, len(score))
			print("k   before",k, len(k))
			
			k = k[self.label[i]]
			print("k   after",k)
			exit()
			tower_logits.append(logits[k])

		#return torch.cat(tower_logits, 0)
		return torch.stack(tower_logits)
	def test(self, x):
		tower_score = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]]
			logits = self.get_logits(sen_matrix)
			score = F.softmax(logits, 1)
			score, _ = torch.max(score, 0)
			tower_score.append(score)
		tower_score = torch.stack(tower_score)
		return list(tower_score.data.cpu().numpy())

class Maxium(Selector):
	def forward(self, x):
		tower_repre = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i+ 1]]
			final_repre, _ = torch.max(sen_matrix, 0)
			tower_repre.append(final_repre)
		stack_repre = torch.stack(tower_repre)
		stack_repre = self.dropout(stack_repre)
		logits = self.get_logits(stack_repre)
		return stack_repre, logits


	def test(self, x):
		tower_repre = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]]
			final_repre, _ = torch.max(sen_matrix, 0)
			tower_repre.append(final_repre)
		stack_repre = torch.stack(tower_repre)
		logits = self.get_logits(stack_repre)
		score = torch.sigmoid(logits)
		# print("score", score, score.size())
		return list(score.data.cpu().numpy())

class Average(Selector):
	def forward(self, x):
		tower_repre = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i+ 1]]
			final_repre = torch.mean(sen_matrix, 0)
			tower_repre.append(final_repre)
		stack_repre = torch.stack(tower_repre)
		stack_repre = self.dropout(stack_repre)
		logits = self.get_logits(stack_repre)
		return logits
	def test(self, x):
		tower_repre = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]]
			final_repre = torch.mean(sen_matrix, 0)
			tower_repre.append(final_repre)
		stack_repre = torch.stack(tower_repre)
		logits = self.get_logits(stack_repre)
		score = F.softmax(logits, 1)
		return list(score.data.cpu().numpy())
