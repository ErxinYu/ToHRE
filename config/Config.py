#coding:utf-8
import torch
import torch.nn as nn
#from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import time
import datetime
import json
import sys
import sklearn.metrics
from tqdm import tqdm
from sklearn.metrics import f1_score
from collections import defaultdict, Counter

def to_var(x):
	return  (torch.from_numpy(x)).cuda()

class Accuracy(object):
	def __init__(self):
		self.correct = 0
		self.total = 0
	def add(self, is_correct):
		self.total += 1
		if is_correct:
			self.correct += 1
	def get(self):
		if self.total == 0:
			return 0.0
		else:
			return float(self.correct) / self.total
	def clear(self):
		self.correct = 0
		self.total = 0 

class Config(object):
	def __init__(self):
		self.acc_NA_global = Accuracy()
		self.acc_not_NA_global = Accuracy()
		self.acc_total_global = Accuracy()
		self.data_path = './data/57K-unjoined'
		self.use_bag = True
		self.use_gpu = True
		self.max_length = 120
		self.pos_num = 2 * self.max_length
		self.flat_num_classes = 53
		self.hidden_size = 300
		self.da = 30
		self.pos_size = 5
		self.max_epoch = 200
		self.opt_method = 'SGD'
		self.optimizer = None
		self.base_model_lr =  0.5
		self.base_model_weight_decay = 1e-5
		self.base_model_drop_prob = 0.5
		self.checkpoint_dir = './checkpoint'
		self.test_result_dir = './test_result'
		self.save_epoch = 1
		self.test_epoch = 1
		self.pretrain_model = None
		self.trainModel = None
		self.testModel = None
		self.train_batch_size = 160
		self.test_batch_size = 160 #test_batch为4的时候, 最大5个: 431 300以上都会错
		self.word_size = 50
		self.window_size = 3
		self.epoch_range = None
	
		##new
		self.acc_NA_local = Accuracy()
		self.acc_not_NA_local_layer0 = Accuracy()
		self.acc_not_NA_local_layer1 = Accuracy()
		self.acc_not_NA_local_layer2 = Accuracy()
		self.acc_total_local = Accuracy()

		self.l1_size = 300
		self.class_embed_size = 50
		self.use_l2 = True
		self.policy_lr = 0.5
		self.policy_weight_decay = 1e-5
		self.n_layers = 3
		self.policy_drop_prob = 0.5
		self.cur_layer = 0
		self.local_loss = 0
		self.predict_label2num = defaultdict(int)
		self.pred_not_na = 0
		self.global_num_classes = 95

		self.is_training = True
		self.flat_probs_only = False
		self.global_ratio = 0
		self.use_label_weight = False
		self.out_model_name = "HRE_57K-unjoined"
		self.gpu = "1"


		print("-------config--------")
		print("self.is_training", self.is_training)
		print("self.policy_lr", self.policy_lr)
		print("use dataset 520K", self.data_path)
		print("self.hidden_size", self.hidden_size)
		print("self.use_label_weight", self.use_label_weight)
		print("self.gpu", self.gpu, "\n\n")

	def set_train_model(self, model):
		self.trainModel = model			 
	def set_test_model(self, model):
		self.testModel = model
	def load_train_data(self):
		print("Reading training data...")
		self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
		self.data_train_word = np.load(os.path.join(self.data_path, 'train_word.npy'))
		self.data_train_pos1 = np.load(os.path.join(self.data_path, 'train_pos1.npy'))
		self.data_train_pos2 = np.load(os.path.join(self.data_path, 'train_pos2.npy'))
		self.data_train_mask = np.load(os.path.join(self.data_path, 'train_mask.npy'))
		self.data_query_label = np.load(os.path.join(self.data_path, 'train_ins_label.npy'))
		self.data_train_label = np.load(os.path.join(self.data_path, 'train_bag_label.npy'))
		self.data_train_scope = np.load(os.path.join(self.data_path, 'train_bag_scope.npy'))
		self.train_hierarchical_ins_label = np.load(os.path.join(self.data_path, "train_hierarchical_ins_label.npy"))
		self.train_hierarchical_bag_label = np.load(os.path.join(self.data_path, "train_hierarchical_bag_label.npy"))
		self.train_order = list(range(len(self.data_train_label)))
		self.train_batches = len(self.data_train_label) / self.train_batch_size
		if len(self.data_train_label) % self.train_batch_size != 0:
			self.train_batches += 1
		self.train_batches = int(self.train_batches)	
		print(len(self.train_hierarchical_bag_label), self.train_hierarchical_bag_label[0])
		print("Finish reading training data")
		self.label_weight_all = defaultdict(int)
		self.label_weight_one = defaultdict(int)

		for index, h_bag_label in enumerate(self.train_hierarchical_bag_label):
			for layer in range(3):
				self.label_weight_all[h_bag_label[layer]] += 1
				if 1 in h_bag_label:
					break
		print(self.label_weight_all)


		for i in self.label_weight_all:
			self.label_weight_all[i] = 1 / (self.label_weight_all[i] ** (0.05))

		self.label_weight = []
		for i in range(95):
			if i not in self.label_weight_all:
				self.label_weight_all[i] = 0
			self.label_weight.append(self.label_weight_all[i])
		self.label_weight = torch.from_numpy(np.array(self.label_weight)).cuda().float()
		print("-"*20, self.label_weight_all[0], "-"*20)
		print(self.label_weight_all)
		print(self.label_weight)



	def load_test_data(self):
		print("Reading testing data...")
		self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
		self.data_test_word = np.load(os.path.join(self.data_path, 'test_word.npy'))
		self.data_test_pos1 = np.load(os.path.join(self.data_path, 'test_pos1.npy'))
		self.data_test_pos2 = np.load(os.path.join(self.data_path, 'test_pos2.npy'))
		self.data_test_mask = np.load(os.path.join(self.data_path, 'test_mask.npy'))
		self.data_test_label = np.load(os.path.join(self.data_path, 'test_bag_label.npy'))
		self.data_test_scope = np.load(os.path.join(self.data_path, 'test_bag_scope.npy'))
		self.test_order = list(range(len(self.data_test_label)))
		self.test_batches = len(self.data_test_label) / self.test_batch_size

		with open(self.data_path + "/test_bag_key.json", "r") as file:
			self.test_bag_key_dict_temp = json.load(file)
			self.test_bag_key_dict = {}
			for bag_id in self.test_bag_key_dict_temp:
				self.test_bag_key_dict[int(bag_id)] = self.test_bag_key_dict_temp[bag_id]
	
		with open(self.data_path + "/relation_id2h_relation_id.json", "r") as file:
			self.relation_id2h_relation_id = json.load(file)
		self.relation_id2h_relation_id_list = []

		self.test_entity_triple = {}
		for bag_id in self.test_bag_key_dict:
			self.test_entity_triple[tuple(self.test_bag_key_dict[bag_id])] = 0
		self.test_batch_attention_query = []
		for relation_id in range(53):
			self.test_batch_attention_query.append(self.relation_id2h_relation_id[str(relation_id)])
		self.test_batch_attention_query = np.array(self.test_batch_attention_query, dtype = np.int64)
		
		if len(self.data_test_label) % self.test_batch_size != 0:
			self.test_batches += 1
		print("data_test_label", len(self.data_test_label))
		self.total_recall = len(np.nonzero(self.data_test_label)[0])
		self.test_batches = int(self.test_batches)
		print("Finish reading testing data")

	def get_train_batch(self, batch):
		input_scope = np.take(self.data_train_scope, self.train_order[batch * self.train_batch_size : (batch + 1) * self.train_batch_size], axis = 0)
		index = []
		scope = [0]
		for num in input_scope:
			index = index + list(range(num[0], num[1] + 1))
			scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
		#print("index", len(index))
		self.batch_scope = scope
		self.batch_word = self.data_train_word[index, :]
		self.batch_pos1 = self.data_train_pos1[index, :]
		self.batch_pos2 = self.data_train_pos2[index, :]
		self.batch_mask = self.data_train_mask[index, :]
		self.batch_attention_query = self.train_hierarchical_ins_label[index, :]
		self.bag_ids = self.train_order[batch * self.train_batch_size : (batch + 1) * self.train_batch_size]	
		self.batch_label = np.take(self.data_train_label, self.train_order[batch * self.train_batch_size : (batch + 1) * self.train_batch_size], axis = 0)
		return len(index)

	def get_test_batch(self, batch):
		input_scope = self.data_test_scope[batch * self.test_batch_size : (batch + 1) * self.test_batch_size]
		index = []
		scope = [0]
		for num in input_scope:
			index = index + list(range(num[0], num[1] + 1))
			scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
		#print("index", len(index))
		self.batch_word = self.data_test_word[index, :]
		self.batch_pos1 = self.data_test_pos1[index, :]
		self.batch_pos2 = self.data_test_pos2[index, :]
		self.batch_mask = self.data_test_mask[index, :]
		self.batch_scope = scope
		self.bag_ids = self.test_order[batch * self.test_batch_size : (batch + 1) * self.test_batch_size]	
		return len(index)	
	def train_one_step(self):
		self.trainModel.embedding.word = to_var(self.batch_word)
		self.trainModel.embedding.pos1 = to_var(self.batch_pos1)
		self.trainModel.embedding.pos2 = to_var(self.batch_pos2)
		self.trainModel.encoder.mask = to_var(self.batch_mask)
		self.trainModel.encoder.bag_ids = self.bag_ids
		self.trainModel.selector.scope = self.batch_scope
		self.trainModel.selector.attention_query = to_var(self.batch_attention_query)
		self.trainModel.encoder.attention_query = to_var(self.batch_attention_query)

	def test_one_step(self):

		self.testModel.embedding.word = to_var(self.batch_word)
		self.testModel.embedding.pos1 = to_var(self.batch_pos1)
		self.testModel.embedding.pos2 = to_var(self.batch_pos2)
		self.testModel.encoder.mask = to_var(self.batch_mask)
		self.testModel.selector.scope = self.batch_scope
		self.testModel.selector.test_attention_query = to_var(self.test_batch_attention_query).long()
		if self.global_ratio >0 :
			return self.testModel.test_flat()

	def test_one_epoch(self):
		print("Test flat model...")
		test_score = []
		for batch in tqdm(range(self.test_batches)):
			self.get_test_batch(batch)
			batch_score = self.test_one_step()
			test_score = test_score + batch_score
		test_result = []
		for i in range(len(test_score)):
			for j in range(1, len(test_score[i])):
				tup = (self.test_bag_key_dict[i][0], self.test_bag_key_dict[i][1], j)
				ans = int(tup in self.test_entity_triple)
				test_result.append([ans, test_score[i][j]])
		test_result = sorted(test_result, key = lambda x: x[1])
		test_result = test_result[::-1]
		pr_x = []
		pr_y = []
		correct = 0
		for i, item in enumerate(test_result):
			correct += item[0]
			pr_y.append(float(correct) / (i + 1))
			pr_x.append(float(correct) / self.total_recall)
		auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
		print("auc_flat: ", auc)
		return auc, pr_x, pr_y
	def test(self):
		best_epoch = None
		best_auc = 0.0
		best_p = None
		best_r = None
		for epoch in self.epoch_range:
			path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
			if not os.path.exists(path):
				continue
			print("Start testing epoch %d" % (epoch))
			self.testModel.load_state_dict(torch.load(path))
			auc, p, r = self.test_one_epoch()
			if auc > best_auc:
				best_auc = auc
				best_epoch = epoch
				best_p = p
				best_r = r
			print("Finish testing epoch %d" % (epoch))
		print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
		print("Storing best result...")
		if not os.path.isdir(self.test_result_dir):
			os.mkdir(self.test_result_dir)
		np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_x.npy'), best_p)
		np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_y.npy'), best_r)
		print("Finish storing")

	













