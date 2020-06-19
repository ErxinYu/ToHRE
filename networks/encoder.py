import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.autograd import Variable
# from gpu_mem_track import MemTracker
# import inspect
# frame = inspect.currentframe()     
# gpu_tracker = MemTracker(frame)     
class _PiecewisePooling_Word_Att(nn.Module):
	def __init(self):
		super(_PiecewisePooling, self).__init__()
	def forward(self, x, mask, hidden_size): #x [244, 3, 300, 120, 1] mask [244,120,3]
		#print("x",x[1][1][1],x.size())
		#print("mask",mask[1], mask.size())
		mask = torch.unsqueeze(mask, 1) # [244,1,120,3]
		#print("mask_1", mask.size())
		mask = torch.unsqueeze(mask, 1) # [244,1,1,120,3]
		#print("mask_2", mask.size())
		#print("mask+x", (mask+x)[1][1][1], (mask+x)[1][1][1], (mask+x).size())
		x, _ = torch.max(mask + x, dim = 3) # mask + x = [244,3,300,120,3].      x = [224,3,300,3]
		#print("x_max", x[0][1], x.size())
		x = x - 100
		#print("x", x.size())
		return x.view(-1, 3, hidden_size * 3) #[224,3,900]

	def test(self, x, mask, hidden_size): #x=[244, 95, 300, 120, 1] mask=[244,120,3]
		#print("x",x.size())
		#print("mask", mask.size())
		mask = torch.unsqueeze(mask, 1) # [244,1,120,3]
		#print("mask_1", mask.size())
		mask = torch.unsqueeze(mask, 1) # [244,1,1,120,3]
		#print("mask_2", mask.size())
		x, _ = torch.max(mask + x, dim = 3) # mask + x = [244,95,300,120,3].      x = [224,95,300,3]
		#print("x_final", x.size())
		x = x - 100
		return x.view(-1, 95, hidden_size * 3) #[224,95,900]
class Word_Att(nn.Module):
	def __init__(self, config):
		super(Word_ATT, self).__init__()
		self.config = config
		self.w1= nn.Embedding(config.da, config.hidden_size) #[150,300]
		self.w2 = nn.Embedding(config.global_num_classes, config.da)#[95,150]
		self.w = nn.Embedding(config.hidden_size, 95) #[300, 95]
		self.activation = nn.Tanh()
		self.activation_1 = nn.ReLU()
		nn.init.xavier_uniform_(self.w1.weight.data)
		nn.init.xavier_uniform_(self.w2.weight.data)
		torch.set_printoptions(precision=8)
	def forward(self, x, attention_query, mask, bag_ids): #x=[241,300,120,1] attention_query=[241,3] mask=[244,120,3]
		#print("x", x[0], x.size())
		#print("attention_query", attention_query.size())
		x_1 = torch.transpose(torch.squeeze(x,3), 1, 2) #[241,120,300]
		# print("x_1", x_1[0], x_1.size())
		# print("mask", mask[0], mask.size())
		a_1 = torch.matmul(x_1, torch.transpose(self.w1.weight, 0, 1))  #[241,120,150] = [241,120,300] * [300, 150] 
		#print("a_1", a_1.size())
		a_2 = self.activation(a_1) #[241, 120, 150]
		a_2 = torch.matmul(a_1, torch.transpose(self.w2.weight, 0, 1))  #[241,120,95] = [241,120,150] * [150,95]
		#print("a_2", a_2.size())

		# a_1 = torch.matmul(x_1, self.w.weight) #[241, 120, 95] = [241,120,300] * [300, 95]

		# a_2 = self.activation(a_1) #[241, 120, 95]

		a_3 = torch.transpose(a_2, 1, 2) #[241,95,120]
		# print("\na_3[0][1]", a_3[0][1], a_3.size())
		# print("\na_3[0][6]", a_3[0][6], a_3.size())
		# a_4 = self.activation_1(a_3) #[241,95,120]
		# print("a_4", a_4[0][5], a_4.size())

		a_4 = F.softmax(a_3, 2) #[241,95,120]

		a_5 = a_4.unsqueeze(3)#[241,95,120,1]
		#print("a_5",a_5[0] a_5.size())

		a_6 = a_5.expand(-1, 95, 120, self.config.hidden_size) #[241,95,120,300]
		#print("a_6", a_6[0][1], a_6.size())

		indices = attention_query #[241,3]
		#print("indices_1", indices[0], indices.size())
		indices = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), a_6.size(2)) #[241,3,120]
		#print("indices_2", indices.size())
		indices = indices.unsqueeze(3).expand(indices.size(0), indices.size(1), a_6.size(2), a_6.size(3))  #[241,3,120,300]
		#print("indices_3", indices.size())
		a_7 = torch.gather(a_6, 1, indices) #[241,3,120,300]  from [241,95,120,300] select [241,3,120,300] need test
		#print("a_7", a_7[0][0], a_7[0][0], a_7.size())
		word_repre = x_1.unsqueeze(1) #[241,1,120,300]
		#print("word_repre", word_repre.size())

		word_repre_1 = word_repre.expand(x_1.size(0), attention_query.size(1), x_1.size(1), x_1.size(2)) #[241,3,120,300]
		#print("word_repre_1", word_repre_1.size())

		word_repre_2 = word_repre_1.mul(a_7) #[241,3,120,300] = [241,3,120,300] * [241,3,120,300]
		#print("word_repre_2", word_repre_2.size())

		word_repre_3 = torch.transpose(word_repre_2, 2, 3) #[241,3,300,120]
		#print("word_repre_3", word_repre_3.size())

		word_repre_final = word_repre_3.unsqueeze(4)# #[241,3,300,120,1]
		#print("word_repre_final", word_repre_final.size())	
		return word_repre_final
	def test(self, x, mask): #x = [241,300,120,1]
		#print("x", x.size())
		
		x_1 = torch.transpose(torch.squeeze(x,3), 1, 2) #[241,120,300]
		#print("x_1", x_1[1], x_1.size())
		#print("mask",mask.size())

		a_1 = torch.matmul(x_1, torch.transpose(self.w1.weight, 0, 1))  #[241,120,300] * [300, 150]   = [241,120,150]
		#print("a_1", a_1.size())
		a_2 = self.activation(a_1) #[241, 120, 150]
		a_2 = torch.matmul(a_1, torch.transpose(self.w2.weight, 0, 1))  #[241,120,95] = [241,120,150] * [150, 95]
		#print("a_2", a_2[2][], a_2.size())

		#a_1 = torch.matmul(x_1, self.w.weight) #[241, 120, 95] = [241,120,300] * [300, 95]
		#print("a_1", a_1.size())

		#a_2 = self.activation(a_1) #[241, 120, 95]

		a_3 = torch.transpose(a_2, 1, 2) #[241,95,120]
		print("a_3", a_3[1][1], a_3.size())

		a_4 = F.log_softmax(a_3, 2) #[241,95,120]
		print("a_4", a_4[1][1], a_4.size())
		exit()
		a_5 = a_4.unsqueeze(3)#[241,95,120,1]
		#print("a_5", a_5.size())

		a_6 = a_5.expand(-1, 95, 120, self.config.hidden_size) #[241,95,120,300]
		#print("a_6", a_6[1][1], a_6.size())

		word_repre = x_1.unsqueeze(1) #[241,1,120,300]
		#print("word_repre", word_repre.size())

		word_repre_1 = word_repre.expand(x_1.size(0), 95, x_1.size(1), x_1.size(2)) #[241,95,120,300]
		#print("word_repre_1", word_repre_1.size())

		word_repre_2 = word_repre_1.mul(a_6) #[241,95,120,300] = [241,95,120,300] * [241,95,120,300]
		#print("x_1", x_1[1], x_1.size())
		#print("word_repre_2\n\n", word_repre_2[1][1], word_repre_2.size())

		word_repre_3 = torch.transpose(word_repre_2, 2, 3) #[241,95,300,120]
		#print("word_repre_3", word_repre_3.size())

		word_repre_final = word_repre_3.unsqueeze(4)# #[241,95,300,120,1]
		#print("word_repre_final", word_repre_final.size())	

		#exit()
		
		return word_repre_final
class PCNN_word_att(nn.Module):
	def __init__(self, config): #
		super(PCNN_word_att, self).__init__()
		self.config = config
		self.mask = None
		self.cnn = _CNN(config)
		self.word_att = Word_ATT(config)
		self.pooling = _PiecewisePooling()
		self.pooling_ori = _PiecewisePooling_ori()
		self.activation = nn.ReLU()
		self.attention_query = None
		self.bag_ids = None
	def forward(self, embedding):
		embedding = torch.unsqueeze(embedding, dim = 1) #[241,1,120,60]
		x = self.cnn(embedding) #[241,300,120,1]
		if self.config.is_training:
			#x = self.word_att(x, self.attention_query, self.mask,self.bag_ids)   #[241, 3, 300, 120, 1]
			x = self.pooling_ori(x, self.mask, self.config.hidden_size) #[241,3,690]
		else:
			# x = self.word_att.test(x, self.mask)   #[241, 95, 300, 120, 1]
			# x = self.pooling.test(x, self.mask, self.config.hidden_size) #[241,95,690]
			x = self.pooling_ori(x, self.mask, self.config.hidden_size)
		return self.activation(x)


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
	def forward(self, embedding):
		embedding = torch.unsqueeze(embedding, dim = 1) #[241,1,120,60]
		x = self.cnn(embedding) #[241,300,120,1]
		# print("encoder cnn x", x.size())
		x = self.pooling(x, self.mask, self.config.hidden_size) #[241, 690]
		# print("encoder pooling x", x.size())
		return self.activation(x)

class _CNN(nn.Module):
	def __init__(self, config):
		super(_CNN, self).__init__()
		self.config = config
		self.in_channels = 1
		self.in_height = self.config.max_length
		self.in_width = self.config.word_size + 2 * self.config.pos_size
		self.kernel_size = (self.config.window_size, self.in_width)
		self.out_channels = self.config.hidden_size
		self.stride = (1, 1)
		self.padding = (1, 0)
		self.cnn = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
	def forward(self, embedding):
		return self.cnn(embedding)
class Word_ATT_v1(nn.Module):
	def __init__(self, config):
		super(Word_ATT, self).__init__()
		self.config = config
		self.word_att_matrix = nn.Embedding(config.hidden_size, config.hidden_size) #[300,300]
		self.relation_vector = nn.Parameter(torch.Tensor(config.hidden_size)) #[300,1]
		# self.relationMatrix = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
		# self.relationVector = nn.Linear(config.hidden_size, 1, bias=False)
		nn.init.xavier_uniform_(self.word_att_matrix.weight.data)
		nn.init.normal_(self.relation_vector)
		torch.set_printoptions(precision=8)
	def forward(self, x): #[241,300,120,1]
		#print("x",x.size())

		x_1 = torch.squeeze(x) #[241,300,120]
		#print("x_1",x_1,x_1.size())

		x_2 = torch.transpose(x_1, 1, 2) #[241,120,300]
		#print("x_2",x_2[0],x_2.size())

		u_1 = torch.matmul(x_2, self.word_att_matrix.weight)  #[241,120,300] *[300,300] = [241,120,300]
		#print("u_1",u_1,u_1.size())
		# u_1 = self.relationMatrix(x_2) #[241,120,300] *[300,300] = [241,120,300]
		# print("u_1",u_1.size())

		u_2 = torch.matmul(u_1, self.relation_vector)  #[241,120,300] * [300,1] = [241,120,1]
		#print("u_2",u_2,u_2.size())
		# u_2 = self.relationVector(u_1)  #[241,120,300] * [300,1] = [241,120,1]
		# print("u_2",u_2.size())

		a_1 = torch.squeeze(u_2) #[241,120]
		#print("a_1",a_1.size())

		a_2 = F.softmax(a_1, 1) #[241,120]
		#print("a_2",torch.sum(a_2[0]),a_2.size())

		a_3 = a_2.unsqueeze(2) #[241,120,1]
		#print("a_3",a_3.size())

		a_4 = a_3.expand(-1, -1, self.config.hidden_size) #[241,120,300]
		#print("a_4",a_4.size())

		word_repre_1 = x_2.mul(a_4) #[241,120,300]
		#print("word_repre_1", word_repre_1.size())

		word_repre_2 = torch.transpose(word_repre_1, 1, 2) #[241,300,120]
		#print("word_repre_2", word_repre_2.size())

		word_repre_final = word_repre_2.unsqueeze(3)# [241,300,120,1]
		#print("word_repre_final", word_repre_final.size())
		#exit()

		return word_repre_final
class _MaxPooling(nn.Module):
	def __init__(self):
		super(_MaxPooling, self).__init__()
	def forward(self, x, hidden_size):
		x, _ = torch.max(x, dim = 2)
		return x.view(-1, hidden_size)
class CNN(nn.Module):
	def __init__(self, config):
		super(CNN, self).__init__()
		self.config = config
		self.cnn = _CNN(config)
		self.pooling = _MaxPooling()
		self.activation = nn.ReLU()
	def forward(self, embedding):
		embedding = torch.unsqueeze(embedding, dim = 1)
		x = self.cnn(embedding)
		x = self.pooling(x, self.config.hidden_size)
		return self.activation(x)	
class _PiecewisePooling_v1(nn.Module):
	def __init(self):
		super(_PiecewisePooling, self).__init__()
	def forward(self, x, mask, hidden_size, global_num_classes): #x [244, 95, 300, 120, 1] 
		print("x",x.size())
		print("mask", mask.size())
		mask = torch.unsqueeze(mask, 1) # [244,1,120,3]
		print("mask_1", mask.size())
		mask = torch.unsqueeze(mask, 1) # [244,1,1,120,3]
		print("mask_2", mask.size())
		print("x", x.size())
		mask_1 = mask.view(-1,3)
		x_1 = mask.view(-1,3)
		k = mask_1 + x_2
		x, _ = torch.max(mask + x, dim = 3) # mask + x = [244,95,300,120,3].      x = [224,95,300,3]
		print("x_final", x.size())
		x = x - 100
		#print("x", x[0], x.size())
		#print("global_num_classes", global_num_classes)

		return x.view(-1, global_num_classes, hidden_size * 3) #[224,95,900]
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