import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
#np.set_printoptions(threshold=np.inf) 

class Policy(nn.Module):
    def __init__(self, conf, n_class, base_model):#in_dim应该是hidden_size *3+class_embedding_size
        super(Policy, self).__init__()
        #self.args = args
        self.conf = conf
        self.class_embed = nn.Embedding(n_class, self.conf.class_embed_size) #
        self.class_embed_bias = nn.Embedding(n_class, 1)
        #self.class_embed_bias = nn.Parameter(torch.Tensor(n_class))
        #这里是对标签嵌入做一些操作 没看懂。
        # stdv = 1. / np.sqrt(self.class_embed.weight.size(1))
        # self.class_embed.weight.data.uniform_(-stdv, stdv)
        # self.class_embed_bias.weight.data.uniform_(-stdv, stdv)
        
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
        in_dim = self.conf.class_embed_size + self.conf.hidden_size * 3 # i 我们的indim = indim_ori(690) + class_embed_size(50) = 740

        self.l1 = nn.Linear(in_dim, self.conf.l1_size)# ours: [740,l1_size]
        self.l2 = nn.Linear(self.conf.l1_size, self.conf.class_embed_size) #[li_size,50]
        if not self.conf.use_l2:
            self.l1 = nn.Linear(in_dim, conf.class_embed_size)# [790,100]
        self.dropout = nn.Dropout(conf.policy_drop_prob)
        self.criterion = torch.nn.CrossEntropyLoss(reduce = False)
        self.criterion_all = torch.nn.CrossEntropyLoss()

    def forward(self, cur_class_batch, next_classes_batch):#详细介绍
        #输入这个批次的现在的标签，以及能产生的所有标签，state_embed在一个batch内是一样的
        #输出就是给出能产生所有标签的概率
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
        probs = torch.bmm(next_classes_embed, h2).squeeze(-1) + nb #[batch,8]     [batch,8,50](8是当前批次子标签最大个数)  [batch,8,1] == [batch, 8 ,1].squeeze(-1)
        return probs
    
    def duplicate_bag_vec(self, indices):#输入bathch中的id，返回bag_vec
        # if self.conf.cur_layer == 1:
        #     print("self.bag_vec_layer1_before", self.bag_vec_layer1[0], self.bag_vec_layer1.size())
    
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

    def generate_logits(self, conf, cur_class_batch, next_classes_batch):#输入[batch,tokens],[batch,cur_class_batch],[batch,max_choices],返回[batch,max_choices_logits]
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
        # print("cur_layer", conf.cur_layer)
        # print("step_sl: cur_class_batch", cur_class_batch, len(cur_class_batch))
        # print("step_sl: next_classes_batch", next_classes_batch, len(next_classes_batch), next_classes_batch[0])
        # print("step_sl: next_classes_batch_true", next_classes_batch_true, len(next_classes_batch_true))
        
        probs = self.generate_logits(conf, cur_class_batch, next_classes_batch) #[160,mc]
        # print("step_sl: probs", probs, probs.size())
        # if conf.cur_layer == 0:
        #     indices_layer0 = Variable(torch.from_numpy(next_classes_batch_true)).long().cuda() #[160]
        #     dummy = indices_layer0.unsqueeze(1).expand(indices_layer0.size(0), probs.size(1)) #[160, mc]
        #     self.layer0_prob = torch.gather(probs, 1, dummy) #[160,mc]       [160,mc] [160, mc]
        #     # print("self.layer0_prob", self.layer0_prob, self.layer0_prob.size())
        #     self.layer0_prob = self.layer0_prob[:,0]
        #     # print("self.layer0_prob", self.layer0_prob, self.layer0_prob.size())


        # elif conf.cur_layer == 1:
        #     #选择上次
        #     self.layer0_prob = self.layer0_prob[indices]
        #     # print("self.layer0_prob", self.layer0_prob, self.layer0_prob.size())
        #     self.layer0_prob = self.layer0_prob.unsqueeze(1).expand(probs.size(0), probs.size(1)) #[15, mc] 与prob一样
        #     # print("self.layer0_prob", self.layer0_prob, self.layer0_prob.size())

        #     #标这次
        #     indices_layer1 = Variable(torch.from_numpy(next_classes_batch_true)).long().cuda() #[15]
        #     dummy = indices_layer1.unsqueeze(1).expand(indices_layer1.size(0), probs.size(1)) #[15, mc]
        #     self.layer1_prob = torch.gather(probs, 1, dummy) #[15,mc]       [15,mc] [15, mc]
        #     # print("self.layer1_prob", self.layer1_prob, self.layer1_prob.size())
        #     probs = probs.mul(self.layer0_prob) #[15,mc]
        #     # print("probs", probs, probs.size())
        #     self.layer1_prob = self.layer1_prob.mul(self.layer0_prob) #[15,mc]
        #     self.layer1_prob = self.layer1_prob[:,0]
        #     # print("self.layer1_prob", self.layer1_prob, self.layer1_prob.size())

        # elif conf.cur_layer == 2:
        #     self.layer1_prob = self.layer1_prob.unsqueeze(1).expand(self.layer1_prob.size(0), probs.size(1))
        #     probs = probs.mul(self.layer1_prob)
        #     # print("probs", probs, probs.size())

        
        # next_classes_batch = Variable(torch.from_numpy(next_classes_batch)).cuda() #[batch,max_choices]
        # probs = (next_classes_batch == 0).float() * -99999 + (next_classes_batch != 0).float() * probs
        # print("probs", probs, probs.size())
        # print("\n\n")
        if next_classes_batch_true is not None:
            cur_size = len(next_classes_batch_true)
            cur_batch_size = len(conf.bag_ids)
            #print("next_classes_batch", next_classes_batch, len(next_classes_batch))
            #print("next_classes_batch_true", next_classes_batch_true, len(next_classes_batch_true))
            next_classes_batch_true_label = next_classes_batch[range(len(next_classes_batch)), next_classes_batch_true] # 真正的label，用来取lable weight
            weight = conf.label_weight[next_classes_batch_true_label]
            #weight = torch.from_numpy(weight).cuda().long()
            #print("next_classes_batch_label", next_classes_batch_label, len(next_classes_batch_label))
            # print("weight", weight, len(weight))
            # print("cur_size", cur_size, self.conf.cur_layer)
            # print("cur_batch_size", cur_batch_size, self.conf.cur_layer)
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
                # print("y_true", y_true, y_true.size())
                # print("probs", probs, probs.size())
                # print("torch.softmax(probs, dim=1)", torch.softmax(probs, dim=1), torch.softmax(probs, dim=1).size())
                # loss = self.criterion(probs, y_true)
                # print("loss", loss, loss.size())
                # loss = self.criterion_all(probs, y_true)
                # print("loss", loss, loss.size())

                loss = self.criterion_all(probs, y_true)
                self.sl_loss += (loss * cur_size / cur_batch_size)
                
                # if conf.cur_layer == 0:  
                #     self.sl_loss += 0.5 * (loss * cur_size / cur_batch_size)
                # else :
                #     self.sl_loss += (loss * cur_size / cur_batch_size)

                #self.sl_loss += loss

        return torch.softmax(probs, dim=1)

    def forward_test(self, cur_class_batch, next_classes_batch):#详细介绍
        #输入这个批次的现在的标签，以及能产生的所有标签，state_embed在一个batch内是一样的
        #输出就是给出能产生所有标签的概率
        cur_class_embed = self.class_embed(cur_class_batch)  # (batch, 50)
        #print("cur_class_embed_before:", cur_class_embed, cur_class_embed.size())
        cur_class_embed = cur_class_embed.unsqueeze(1).expand(cur_class_embed.size(0), next_classes_batch.size(1), cur_class_embed.size(1)) 
                             #[batch, mc, 50]
        #print("cur_class_embed_after:", cur_class_embed, cur_class_embed.size())
        next_classes_embed = self.class_embed(next_classes_batch)  # (batch, mc, 50)
        nb = self.class_embed_bias(next_classes_batch)
        #print("nb_before:", nb, nb.size())
        #nb = nb.expand(nb.size(0), nb.size(1), nb.size(1))
        #print("nb_after:", nb, nb.size())
        states_embed = self.bag_vec   #[batch, mc, 690]

        
        
        # print("states_embed_before", states_embed, states_embed.size())
        # print("next_classes_embed", next_classes_embed.size())
        states_embed = torch.cat((states_embed, cur_class_embed), 2) #[batch,mc,740]
        states_embed = self.dropout(states_embed) #[batch,mc,740]
        # print("states_embed_after", states_embed, states_embed.size())
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

    def get_test_bag_vec(self, next_classes_batch, indices):#输入bathch中的id，返回bag_vec

        #input: self.bag_vec_test-[160,95,690]    next_classes_batch-[160,8]    #indices [160] [1280] [5440]    
        #output: self.bac_vec-[160,8,690]


        self.bag_vec_test = self.bag_vec_test[indices] #[160,95,690]
        # print("next_classes_batch", next_classes_batch, len(next_classes_batch), "cur_layer", self.conf.cur_layer)
        # print("self.bag_vec_test", self.bag_vec_test[0], self.bag_vec_test.size())
        indices = torch.from_numpy(next_classes_batch).long().cuda() #[160,8]
        dummy = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), self.bag_vec_test.size(2)) 
        # print("dummy", dummy, dummy.size()) #[batch_selected, mc, 690]
        self.bag_vec = torch.gather(self.bag_vec_test, 1, dummy) #[160,8,690]       [160,mc,690] [160, mc, 690]
        #print("bag_vec", self.bag_vec.size())
        #print("self.bag_vec_test[1,1,:]",self.bag_vec_test[2,3,:])
        # print("self.bag_vec_test", self.bag_vec_test, self.bag_vec_test.size())
        # print("indices", indices, indices.size())
        #print("self.bag_vec[1,1,:]",self.bag_vec[2,3,:])






