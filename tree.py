import logging
import pickle
from collections import defaultdict, Counter


import torch
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

import json


class Tree:
    def __init__(self, conf):

        self.conf = conf

        with open(self.conf.data_path + "/p2c_id.json", "r") as file:
            self.p2c_idx_temp = json.load(file)  

        self.train_hierarchical_bag_label_np = np.load(self.conf.data_path + "/train_hierarchical_bag_label.npy")
        self.test_hierarchical_bag_label_np = np.load(self.conf.data_path + "/test_hierarchical_bag_label.npy")
        self.p2c_idx = {}
        for i in self.p2c_idx_temp:
            self.p2c_idx[int(i)] = self.p2c_idx_temp[i]
        

        not_na_bag = 0
        self.train_label2num = defaultdict(int) 
        self.train_hierarchical_bag_label = {}   
        for bag_id in range(len(self.train_hierarchical_bag_label_np)):
            self.train_hierarchical_bag_label[bag_id] = self.train_hierarchical_bag_label_np[bag_id]




        self.test_hierarchical_bag_label = {}  
        self.test_label2num = defaultdict(int) 
        not_na_bag = 0 
        for bag_id in range(len(self.test_hierarchical_bag_label_np)):
            self.test_hierarchical_bag_label[bag_id] = self.test_hierarchical_bag_label_np[bag_id]


        with open(self.conf.data_path + "/test_hierarchical_bag_multi_label.json", "r") as file:
            self.test_hierarchical_bag_multi_label_temp = json.load(file)
            self.test_hierarchical_bag_multi_label = {}
            for bag_id in self.test_hierarchical_bag_multi_label_temp:
                self.test_hierarchical_bag_multi_label[int(bag_id)] = self.test_hierarchical_bag_multi_label_temp[bag_id] 
        
        with open(self.conf.data_path + "/relation_id2h_relation_id.json", "r") as file:
            self.relation_id2h_relation_id = json.load(file)

        self.next_true_bin, self.next_true = self.generate_next_true()
        self.p2c_idx_np = self.pad_p2c_idx()
        self.n_class = len(self.p2c_idx)
        self.conf.global_num_classes = len(self.p2c_idx)
        self.n_update = 0
        self.cur_epoch = 0

    def pad_p2c_idx(self): #构造p2c_idx_np
        col = max([len(c) for c in self.p2c_idx.values()]) #col是含有子节点最多的个数。 可以直接+1，因为这里的allow_up没用了
        res = np.zeros((len(self.p2c_idx), col), dtype=int) #[104,25]
        for row_i in range(len(self.p2c_idx)):
            res[row_i, :len(self.p2c_idx[row_i])] = self.p2c_idx[row_i]
        return res

    def p2c_batch(self, ids):#输入标签id，返回标签id的子标签+标签id
        # ids is virtual
        res = self.p2c_idx_np[ids]
        # 去掉都是0的列
        return res[:, ~np.all(res == 0, axis=0)]

    def generate_next_true(self):#构造
        next_true_bin = defaultdict(lambda: defaultdict(list))
        next_true = defaultdict(lambda: defaultdict(list))

        for did in range(len(self.train_hierarchical_bag_label)):
            class_idx_set = set(self.train_hierarchical_bag_label[did]) #是当前文档标签的集合
            class_idx_set.add(0)
            for c in class_idx_set:
                for idx, next_c in enumerate(self.p2c_idx[c]):
                    if next_c in class_idx_set:
                        next_true_bin[did][c].append(1)#key是当前文档id，value是一个dict, {"cur_label"：[0，1，0]} 原来的标签的子标签是否在已经在当前标签中
                        next_true[did][c].append(next_c)#key是当前文档id，value是一个dict, {"cur_label"：[34，55]} 如果已经在当前标签中，添加标签的子标签
                    else:
                        next_true_bin[did][c].append(0)
                # if lowest label 我觉得这个可以不用 (在我们的项目中，标签比较均衡，留着好像也没事) 
                if len(next_true[did][c]) == 0:
                    next_true[did][c].append(c)#如果没有子标签在当前文档标签中，就添加当前标签。
        return next_true_bin, next_true#构造

    def get_next(self, cur_class_batch, next_classes_batch, bag_ids):#详细介绍
        assert len(cur_class_batch) == len(bag_ids)
        next_classes_batch_true_bin = np.zeros(next_classes_batch.shape)
        next_class_batch = []
        for ct, (c, did) in enumerate(zip(cur_class_batch, bag_ids)):
            nt = self.next_true_bin[did][c]
            if len(self.next_true[did][c]) == 0:
                print(did)
                exit(-1)   
            next_classes_batch_true_bin[ct][:len(nt)] = nt
            for idx in self.next_true[did][c]:
                next_class_batch.append(idx)
        indices, next_classes_batch_true = np.where(next_classes_batch_true_bin == 1) #挑出batch中 能够进行到下一层的next_classes_batch_true(batch中为0和95的都被过滤掉了)
        next_class_batch = np.array(next_class_batch)[indices]#挑出batch中 能够进行到下一层的next_classes_batch(batch中为1和95的都被过滤掉了)
        bag_ids = [bag_ids[idx] for idx in indices]
        return next_classes_batch_true, indices.tolist(), next_class_batch, bag_ids
        
    def get_next_all(self, cur_class_batch, next_classes_batch, bag_ids):
        assert len(cur_class_batch) == len(bag_ids)
        next_class_batch_pred = []
        indices = []
        for ct, (next_classes, did) in enumerate(zip(next_classes_batch, bag_ids)):
            for next_class in next_classes:
                if next_class != 0:
                    indices.append(ct)
                    next_class_batch_pred.append(next_class)

        return np.array(indices), np.array(next_class_batch_pred)

    def get_next_by_probs(self, conf, cur_class_batch, next_classes_batch, bag_ids, probs, indices, cur_step):
        assert len(cur_class_batch) == len(bag_ids)  == len(probs)
        #print("probs_before:",len(probs),probs)
        #print("tree: indices:", len(indices), indices)
        next_class_batch_pred = []
        preds = (torch.max(probs, dim = 1))[1].cpu().numpy()
        probs = (torch.max(probs, dim = 1))[0].cpu().detach().numpy()
        # print("tree: preds:",len(preds),preds)
        # print("tree: probs_after:", len(probs), probs)
        # exit()
        #print("tree: next_classes_batch", next_classes_batch, len(next_classes_batch))
        for ct, (next_classes, pred) in enumerate(
                zip(next_classes_batch, preds)):
            pred_label = next_classes[pred]
            next_class_batch_pred.append(pred_label)
        return np.array(next_class_batch_pred), probs

#tree = Tree()
# cur_class_batch = [1,1,1,1,1,1,2]
# next_classes_batch = tree.p2c_batch(cur_class_batch)
# print(next_classes_batch)


