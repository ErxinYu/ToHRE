import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
#np.set_printoptions(threshold=np.inf) 

class Policy(nn.Module):
    def __init__(self, conf, n_class, base_model):
        super(Policy, self).__init__()
        self.conf = conf
        self.class_embed = nn.Embedding(n_class, self.conf.class_embed_size) #
        self.class_embed_bias = nn.Embedding(n_class, 1)

        
        nn.init.xavier_uniform_(self.class_embed.weight.data)
        nn.init.normal_(self.class_embed_bias.weight.data)
        self.saved_log_probs = []
        self.rewards = []
        self.rewards_greedy = []
        self.bag_vec = None
        self.bag_vec_test = None
        self.bag_vec_layer0 = None
        self.bag_vec_layer1 = None
        self.bag_vec_layer2 = None
        self.layer0_prob = None
        self.layer1_prob = None
        self.base_model = base_model
        self.sl_loss = 0
        in_dim = self.conf.class_embed_size + self.conf.hidden_size * 3 

        self.l1 = nn.Linear(in_dim, self.conf.l1_size)# 
        self.l2 = nn.Linear(self.conf.l1_size, self.conf.class_embed_size) 
        if not self.conf.use_l2:
            self.l1 = nn.Linear(in_dim, conf.class_embed_size)# 
        self.dropout = nn.Dropout(conf.policy_drop_prob)
        self.criterion = torch.nn.CrossEntropyLoss(reduce = False)
        self.criterion_all = torch.nn.CrossEntropyLoss()

    def forward(self, cur_class_batch, next_classes_batch):#

        cur_class_embed = self.class_embed(cur_class_batch)  # (batch, 50)
        next_classes_embed = self.class_embed(next_classes_batch)  # (batch, mc, 50)
        nb = self.class_embed_bias(next_classes_batch).squeeze(-1)
        states_embed = self.bag_vec   #[batch, 690]

        states_embed = torch.cat((states_embed, cur_class_embed), 1) #[batch. 740]

        states_embed = self.dropout(states_embed)
        if self.conf.use_l2: #true
            h1 = F.relu(self.l1(states_embed)) #  [batch, 300] = [batch, 740] * [740, 300]
            h2 = F.relu(self.l2(h1)) #[batch, 50] = [batch, 300] * [300, 50]
        else:
            h2 = F.relu(self.l1(states_embed))
        h2 = h2.unsqueeze(-1)  # (batch, 50, 1）
        probs = torch.bmm(next_classes_embed, h2).squeeze(-1) + nb #[batch,8]     
        return probs
    
    def duplicate_bag_vec(self, indices):#

    
        self.bag_vec_layer0 = self.bag_vec_layer0[indices]
        self.bag_vec_layer1 = self.bag_vec_layer1[indices]
        self.bag_vec_layer2 = self.bag_vec_layer2[indices]

        if self.conf.cur_layer == 0:
            self.bag_vec = self.bag_vec_layer0
        elif self.conf.cur_layer == 1:
            self.bag_vec = self.bag_vec_layer1
            #print("self.bag_vec_layer1_after", self.bag_vec_layer1[0], self.bag_vec_layer1.size())
        elif self.conf.cur_layer == 2:
            self.bag_vec = self.bag_vec_layer2

    def generate_logits(self, conf, cur_class_batch, next_classes_batch):#
        # print("cur_class_batch")
        # print(cur_class_batch,len(cur_class_batch))
        cur_class_batch = Variable(torch.from_numpy(cur_class_batch)).cuda()
        # print("next_classes_batch")
        # print(next_classes_batch, len(next_classes_batch))
        next_classes_batch = Variable(torch.from_numpy(next_classes_batch)).cuda() #[batch,max_choices]
        probs = self(cur_class_batch, next_classes_batch)
        # mask padding relations
        probs = (next_classes_batch == 0).float() * -99999 + (next_classes_batch != 0).float() * probs
        return probs

    def step_sl(self, conf, cur_class_batch, next_classes_batch, next_classes_batch_true, indices):
        assert len(cur_class_batch) == len(self.bag_vec)

        
        probs = self.generate_logits(conf, cur_class_batch, next_classes_batch) #[160,mc]
        if next_classes_batch_true is not None:
            cur_size = len(next_classes_batch_true)
            cur_batch_size = len(conf.bag_ids)

            next_classes_batch_true_label = next_classes_batch[range(len(next_classes_batch)), next_classes_batch_true]
            weight = conf.label_weight[next_classes_batch_true_label]

            y_true = Variable(torch.from_numpy(next_classes_batch_true)).long().cuda()
            if conf.use_label_weight:
                loss = self.criterion(probs, y_true)
                loss = loss * weight
                loss = torch.sum(loss, dim=0)
                loss = loss/cur_batch_size
                if conf.cur_layer == 0:  
                    self.sl_loss += 2 * loss
                else :
                    self.sl_loss += 0.5 * loss
            else:                


                loss = self.criterion_all(probs, y_true)
                self.sl_loss += (loss * cur_size / cur_batch_size)


        return torch.softmax(probs, dim=1)

    def forward_test(self, cur_class_batch, next_classes_batch):#

        cur_class_embed = self.class_embed(cur_class_batch)  #
        cur_class_embed = cur_class_embed.unsqueeze(1).expand(cur_class_embed.size(0), next_classes_batch.size(1), cur_class_embed.size(1)) 

        next_classes_embed = self.class_embed(next_classes_batch)  #
        nb = self.class_embed_bias(next_classes_batch)

        states_embed = self.bag_vec   #

        
        states_embed = torch.cat((states_embed, cur_class_embed), 2) #[batch,mc,740]
        states_embed = self.dropout(states_embed) #[batch,mc,740]
        if self.conf.use_l2: #true
            h1 = F.relu(self.l1(states_embed)) #  [batch, mc, 300] = [batch,mc,740] * [740, 300]
            h2 = F.relu(self.l2(h1)) # # (batch, mc, 50) = [batch, mc, 300] * [300, 50]
        else:
            h2 = F.relu(self.l1(states_embed))# (batch, mc, 50)
        h2 = h2.permute(0,2,1)
        probs = torch.bmm(next_classes_embed, h2) + nb #[batch,mc,mc]    [batch,8,8] = [batch, mc, 50] * [batch, 50, mc]
        return probs

    def step_sl_test(self, conf, cur_class_batch, next_classes_batch):
        assert len(cur_class_batch) == len(self.bag_vec)
        cur_class_batch = Variable(torch.from_numpy(cur_class_batch)).cuda()
        next_classes_batch = Variable(torch.from_numpy(next_classes_batch)).cuda() #[batch,max_choices]
        probs = self.forward_test(cur_class_batch, next_classes_batch)
        probs = probs.permute(0,2,1)
        probs = F.softmax(probs, 2) 
        probs = torch.diagonal(probs, offset=0, dim1=1, dim2=2) #[batch, 8, 1]

        return probs 

    def get_test_bag_vec(self, next_classes_batch, indices):#



        self.bag_vec_test = self.bag_vec_test[indices] #

        indices = torch.from_numpy(next_classes_batch).long().cuda() #[160,8]
        dummy = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), self.bag_vec_test.size(2)) 
        # print("dummy", dummy, dummy.size()) #[batch_selected, mc, 690]
        self.bag_vec = torch.gather(self.bag_vec_test, 1, dummy) #[160,8,690]       [160,mc,690] [160, mc, 690]






