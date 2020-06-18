import config
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse
from PCNN_ATT import PCNN_ATT
import os
import pickle

from collections import defaultdict, Counter
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Policy
from tree import Tree
import sklearn.metrics

def calc_sl_loss(probs, update=True):
    y_true = conf.batch_label
    y_true = Variable(torch.from_numpy(y_true)).cuda().long()
    loss = criterion(probs, y_true)
    return loss
def forward_step_sl():#详细介绍

    # TODO can reuse logits
    logits_layers, logits_total, flat_probs = policy.base_model()#[64,53] 每个flat关系的概率 here1
    if conf.global_ratio > 0:
        global_loss = calc_sl_loss(flat_probs, update=False)#一个tensor单值 在论文中就是flat_loss
    else:
        flat_probs = None
        global_loss = 0
    
    if conf.flat_probs_only:
        policy.sl_loss = global_loss
        return global_loss, flat_probs

    policy.bag_vec_layer0 = logits_layers[0]
    policy.bag_vec_layer1 = logits_layers[1]
    policy.bag_vec_layer2 = logits_layers[2]
    # policy.bag_vec = logits
    bag_ids = conf.bag_ids
    cur_batch_size = len(bag_ids) #一般都是64
    cur_class_batch = np.zeros(cur_batch_size, dtype=int)
    for layer in range(conf.n_layers):#3
        conf.cur_layer = layer
        #print("cur_class_batch", cur_class_batch, len(cur_class_batch), "layer", layer)
        next_classes_batch = tree.p2c_batch(cur_class_batch)#[batch,上一阶段标签的子标签]，可以看成第n层及他之前的标签
        #print("next_classes_batch", next_classes_batch, next_classes_batch[0], len(next_classes_batch), "layer", layer)
        next_classes_batch_true, indices, next_class_batch, bag_ids = tree.get_next(cur_class_batch, next_classes_batch, bag_ids)# next_class_batch_true和indices都是相对位置
        # print("next_classes_batch_true", next_classes_batch_true, len(next_classes_batch_true), "layer", layer)
        # print("indices", indices, len(indices), "layer", layer)
        # print("next_class_batch", next_class_batch, len(next_class_batch), "layer", layer)
        # print("bag_ids", bag_ids, len(bag_ids), "layer", layer)
        if len(indices) == 0:
           break
        policy.duplicate_bag_vec(indices)
        cur_class_batch = cur_class_batch[indices]
        next_classes_batch = next_classes_batch[indices]
        #print("policy.bag_vec", policy.bag_vec, policy.bag_vec.size(), "layer", layer)
        probs = policy.step_sl(conf, cur_class_batch, next_classes_batch, next_classes_batch_true)#加入local_loss
        cur_class_batch = next_class_batch

        ###cal train step hierarchical
        preds = torch.max(probs, dim = 1)[1].cpu().numpy()
        preds = [next_classes_batch[i][preds[i]] for i in range(len(preds))]
        conf.local_loss = policy.sl_loss
        for i, var in enumerate(indices):
                y_pred = preds[i]
                y_true = tree.train_hierarchical_bag_label[bag_ids[i]]#list which is label
                #print("y_pred", y_pred, "y_true", y_true)
                if y_pred != 1:
                    if layer == 0:
                        conf.predict_label2num[y_pred] += 1
                        conf.pred_not_na += 1
                        #print("y_pred:", y_pred, "y_true:", y_true, "y_true_all", y_true_all)
                        conf.acc_not_NA_local_layer0.add(y_pred in y_true)
                    elif layer == 1:
                        #print("y_pred:", y_pred, "y_true:", y_true, "y_true_all", y_true_all)
                        conf.acc_not_NA_local_layer1.add(y_pred in y_true)
                    elif layer == 2:
                        #print("y_pred:", y_pred, "y_true:", y_true, "y_true_all", y_true_all)
                        conf.acc_not_NA_local_layer2.add(y_pred in y_true)
                elif y_pred == 1:
                    conf.acc_NA_local.add(y_pred in y_true)
    #policy.sl_loss /= conf.n_steps# n_steps=17 不知道这里为什么取17
    policy.sl_loss = (1 - conf.global_ratio) * policy.sl_loss + conf.global_ratio * global_loss
    return global_loss, flat_probs
def cal_train_one_step_flat(probs):
    _, _output = torch.max(probs, dim = 1)
    _output = _output.cpu().numpy()

    for i, prediction in enumerate(_output):
        if conf.batch_label[i] == 0:
            conf.acc_NA_global.add(conf.batch_label[i] == prediction)
        else:
            conf.acc_not_NA_global.add(conf.batch_label[i] == prediction)
        conf.acc_total_global.add(conf.batch_label[i] == prediction)
def train():
    conf.set_train_model(policy.base_model)
    best_auc = 0.0
    best_p = None
    best_r = None
    best_epoch = 0
    num_delete_bag = 0
    #policy.load_state_dict(torch.load("./checkpoint/word_epoch" + str(5)))  
    for epoch in range(1, conf.max_epoch + 1):
        conf.is_training = True
        policy.train()
        print('Epoch ' + str(epoch) + ' starts...')
        loss_total = 0
        np.random.shuffle(conf.train_order)
        #local acc
        conf.acc_NA_local.clear()
        conf.acc_not_NA_local_layer0.clear()
        conf.acc_not_NA_local_layer1.clear()
        conf.acc_not_NA_local_layer2.clear()
        conf.acc_total_local.clear()      
        conf.predict_label2num = defaultdict(int)
        conf.pred_not_na = 0 
        #global acc
        conf.acc_NA_global.clear()
        conf.acc_not_NA_global.clear()
        conf.acc_total_global.clear()

        for batch_num in range(conf.train_batches):
            sen_num = conf.get_train_batch(batch_num)
            # if sen_num > 300: %为了防止包太大搞的
            #     num_delete_bag += 1
            #     print("num_delete_bag-index",num_delete_bag, sen_num)
            #     continue
            conf.train_one_step()
            global_loss, flat_probs = forward_step_sl()

            policy_optimizer.zero_grad()
            policy.sl_loss.backward()
            policy_optimizer.step()

            if conf.flat_probs_only:
                cal_train_one_step_flat(flat_probs)
                sys.stdout.write("Global Information: epoch %d step %d  | loss: %f, NA accuracy: %f, not NA accuracy: %f, total accuracy: %f\r" % (epoch, batch_num, policy.sl_loss, conf.acc_NA_global.get(), conf.acc_not_NA_global.get(), conf.acc_total_global.get()))
            else:
                sys.stdout.write("Local Information: epoch %d step %d | loss: %f, NA acc: %f, layer0 accuracy: %f, layer1 accuracy: %f, layer2 accuracy: %f\r" % (epoch, batch_num, conf.local_loss, conf.acc_NA_local.get(), conf.acc_not_NA_local_layer0.get(), conf.acc_not_NA_local_layer1.get(), conf.acc_not_NA_local_layer2.get()))     
            sys.stdout.flush()
            policy.sl_loss = 0
        print("\ntrain:predict_label2num", conf.predict_label2num, "pred_not_na", conf.pred_not_na)
        if epoch % conf.save_epoch == 0:
            print('Train Epoch ' + str(epoch) + ' has finished')
            test_epoch_by_all(epoch)
            print('Saving model...\n')
            torch.save(policy.state_dict(), "./checkpoint/word_epoch" + str(epoch))
    print("Finish training")
    print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
    print("Storing best result...")
def test_epoch_by_all(epoch):
    
    # set test model
    model_file = "./checkpoint/word_epoch" + str(epoch)
    print('Test local: test_epoch_by_all model  ' + model_file)
    if not conf.is_training:
        policy.load_state_dict(torch.load(model_file))  
    conf.is_training = False
    policy.eval()
    conf.set_test_model(policy.base_model)

    #test local model
    test_result_layer_0 = []
    test_result = []
    bagid_label2prob_dict = defaultdict()
    conf.acc_NA_local.clear()
    conf.acc_not_NA_local_layer0.clear()
    conf.acc_not_NA_local_layer1.clear()
    conf.acc_not_NA_local_layer2.clear()
    conf.acc_total_local.clear()
    
    #test global model for comparation
    if conf.global_ratio > 0 :
        conf.acc_NA_global.clear()
        conf.acc_not_NA_global.clear()
        conf.acc_total_global.clear()
        conf.testModel = policy.base_model
        auc, pr_x, pr_y = conf.test_one_epoch()
        print("auc_flat:", auc)
    if conf.flat_probs_only:
        return

    predict_label2num = defaultdict(int)
    pred_not_na = 0
    over = 0
    for batch_num in tqdm(range(conf.test_batches)):  
        sen_num = conf.get_test_batch(batch_num)
        # if sen_num > 100: 为了防止包太大搞的
        #     print(sen_num)
        #     over += 1
        #     continue
        conf.test_one_step()
        logits = policy.base_model.test_hierarchical() #[160,96,690]
        policy.bag_vec_test = logits
        bag_ids = conf.bag_ids
        cur_batch_size = len(bag_ids)
        cur_class_batch = np.zeros(cur_batch_size, dtype=int)
        indices = torch.from_numpy(np.array(range(len(bag_ids)))).cuda()
        for layer in range(conf.n_layers):#3
            conf.cur_layer = layer
            next_classes_batch = tree.p2c_batch(cur_class_batch)#[batch,mc]
            policy.get_test_bag_vec(next_classes_batch, indices) #根据next_classes_batch选择 self.bag_vec 
            h_probs = policy.step_sl_test(conf, cur_class_batch, next_classes_batch)
            h_probs_np = h_probs.cpu().detach().numpy()                

            for i, var in enumerate(indices):
                y_pred_classes = next_classes_batch[i]
                y_true = tree.test_hierarchical_bag_label[bag_ids[i]] #list which is label
                cur_bag_id = bag_ids[i]
                for j in range(len(y_pred_classes)):
                    y_pred = y_pred_classes[j]
                    if y_pred != 0:
                        bagid_label = str(cur_bag_id) + "_" + str(y_pred)
                        bagid_label2prob_dict[bagid_label] = float(h_probs_np[i][j])
            indices, next_class_batch_pred = tree.get_next_all(cur_class_batch, next_classes_batch, bag_ids)
            if len(indices) == 0:
                break    
            bag_ids = [bag_ids[idx] for idx in indices]
            cur_class_batch = next_class_batch_pred
    print("over", over)
    file_name = "./test_result/word_epoch" + str(epoch) + ".json"
    with open(file_name, "w") as file:
        json.dump(bagid_label2prob_dict, file)
    torch.cuda.empty_cache() #清空显存
    

    #可以选择 注释掉 相等于边训练 边测试
    test_result = []
    error = 0
    for bag_id in tqdm(range(len(tree.test_hierarchical_bag_multi_label))):
        y_true = tree.test_hierarchical_bag_multi_label[bag_id]
        for i in range(1, len(conf.test_batch_attention_query)):

            indices = conf.test_batch_attention_query[i]
            predict_layer_0_index = str(bag_id) + "_" + str(indices[0])
            predict_layer_1_index = str(bag_id) + "_" + str(indices[1])
            predict_layer_2_index = str(bag_id) + "_" + str(indices[2])
            label_layer_0_index = str(bag_id) + "_" + str(y_true[0])
            label_layer_1_index = str(bag_id) + "_" + str(y_true[1])
            label_layer_2_index = str(bag_id) + "_" + str(y_true[2])
            if predict_layer_0_index not in bagid_label2prob_dict:
                predict_layer0_prob = 0
                predict_layer1_prob = 0
                predict_layer2_prob = 0
                predict_layer2_prob = 0
                label_layer0_prob = 0
                label_layer1_prob = 0
                label_layer2_prob = 0
                # print("error",error)
                error += 1

            else:
                predict_layer0_prob = bagid_label2prob_dict[predict_layer_0_index]
                predict_layer1_prob = bagid_label2prob_dict[predict_layer_1_index]
                if indices[1] in [27,34,28, 22, 20, 21, 33,29,31,30,25,24,32,39,40,11,13,14,15,9,10,42,18,27,19,41]:
                    predict_layer2_prob = 1
                else:
                    predict_layer2_prob = bagid_label2prob_dict[predict_layer_2_index]
            
                label_layer0_prob = bagid_label2prob_dict[label_layer_0_index]
                label_layer1_prob = bagid_label2prob_dict[label_layer_1_index]
                label_layer2_prob = bagid_label2prob_dict[label_layer_2_index]

            if predict_layer_2_index in bagid_label2prob_dict:
                predict_prob = predict_layer0_prob * predict_layer1_prob * predict_layer2_prob
                label_prob = label_layer0_prob * label_layer1_prob * label_layer2_prob
                ans = int(indices[2] in y_true)
                test_result.append([ans, predict_prob, indices[2], predict_layer0_prob, predict_layer1_prob, predict_layer2_prob, y_true, label_prob, label_layer0_prob, label_layer1_prob, label_layer2_prob])
            # else:
            #     print(predict_layer_0_index,predict_layer_1_index,predict_layer_2_index)
    test_result = sorted(test_result, key = lambda x: x[1])
    test_result = test_result[::-1]
    pr_x = []
    pr_y = []
    correct = 0
    p_4 = 0
    for i, item in enumerate(test_result):
        correct += item[0]
        pr_y.append(float(correct) / (i + 1))
        pr_x.append(float(correct) / conf.total_recall)
    auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
    for i in range(len(pr_x)): 
        if pr_x[i] >= 0.4:
            print("precision at relll@0.4")
            p_4 = pr_y[i]
            print(pr_x[i])
            print(pr_y[i])
            break     
    print("test auc_local: ", auc)
    print("p_4", p_4)
    return auc, pr_x, pr_y, test_result


def test():
    best_epoch = None
    best_auc = 0.0
    best_p = None
    best_r = None
    best_p_4 = 0
    best_test_result = None

    #epochs = range(1，10)
    epochs = [3,4,5]
    for epoch in epochs:
        auc, p, r, test_result = test_epoch_by_all(epoch) 
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_p = p
            best_r = r
            best_test_result = test_result
        print("Finish testing epoch %d" % (epoch))

    print("Best epoch = %d | auc = %f | p_recall4 = %f | p@100 = %f| P@200 = %f| P@300 = %f | P@1000 = %f | |P@2000 = %f| " % (best_epoch, best_auc, best_p_4, best_r[100], best_r[200], best_r[300], best_r[1000], best_r[2000]))
    print("Storing best result...")
    if not os.path.isdir(conf.test_result_dir):
        os.mkdir(conf.test_result_dir)
    np.save(os.path.join(conf.test_result_dir, 'HRE_wo_weight_x.npy'), best_p)
    np.save(os.path.join(conf.test_result_dir, 'HRE_wo_weight_y.npy'), best_r)

    file_name = "./test_result/epoch_" + str(epoch) + ".txt"
    file_name_1 = "./test_result/epoch_p" + str(best_epoch) + ".txt"
    file_name_2 = "./test_result/epoch_n" + str(best_epoch) + ".txt"
    with open(file_name, "w") as file, open(file_name_1, "w") as file1, open(file_name_2, "w") as file2:
        for i in tqdm(range(len(best_test_result))):
            best_test_result[i].append(i)
            print(best_test_result[i], file = file)
            if best_test_result[i][0] == 1:
                print(best_test_result[i], file = file1)   
            else:
                print(test_result[i], file = file2)
    print("Finish storing")

def test_json(epoch):
    print(len(conf.test_batch_attention_query))
    print("\nstart test epoch %d "%(epoch))
    file_name = "./test_result/bagid_label2prob_lr0.4_wo_weight_epoch" + str(epoch)+ ".json"
    with open(file_name, "r") as file:
        bagid_label2prob_dict = json.load(file)
    print("read file from ", file_name)
    print(len(bagid_label2prob_dict))
    test_result = []
    error = 0
    for bag_id in tqdm(range(len(tree.test_hierarchical_bag_multi_label))):
        y_true = tree.test_hierarchical_bag_multi_label[bag_id]
        for i in range(1, len(conf.test_batch_attention_query)):

            indices = conf.test_batch_attention_query[i]
            predict_layer_0_index = str(bag_id) + "_" + str(indices[0])
            predict_layer_1_index = str(bag_id) + "_" + str(indices[1])
            predict_layer_2_index = str(bag_id) + "_" + str(indices[2])
            label_layer_0_index = str(bag_id) + "_" + str(y_true[0])
            label_layer_1_index = str(bag_id) + "_" + str(y_true[1])
            label_layer_2_index = str(bag_id) + "_" + str(y_true[2])
            if predict_layer_0_index not in bagid_label2prob_dict:
                predict_layer0_prob = 0
                predict_layer1_prob = 0
                predict_layer2_prob = 0
                predict_layer2_prob = 0
                label_layer0_prob = 0
                label_layer1_prob = 0
                label_layer2_prob = 0
                print("error",error)
                error += 1

            else:
                predict_layer0_prob = bagid_label2prob_dict[predict_layer_0_index]
                predict_layer1_prob = bagid_label2prob_dict[predict_layer_1_index]
                if indices[1] in [27,34,28, 22, 20, 21, 33,29,31,30,25,24,32,39,40,11,13,14,15,9,10,42,18,27,19,41]:
                    predict_layer2_prob = 1
                else:
                    predict_layer2_prob = bagid_label2prob_dict[predict_layer_2_index]
            
                label_layer0_prob = bagid_label2prob_dict[label_layer_0_index]
                label_layer1_prob = bagid_label2prob_dict[label_layer_1_index]
                label_layer2_prob = bagid_label2prob_dict[label_layer_2_index]

            if predict_layer_2_index in bagid_label2prob_dict:
                predict_prob = predict_layer0_prob * predict_layer1_prob * predict_layer2_prob
                label_prob = label_layer0_prob * label_layer1_prob * label_layer2_prob
                ans = int(indices[2] in y_true)
                test_result.append([ans, predict_prob, indices[2], predict_layer0_prob, predict_layer1_prob, predict_layer2_prob, y_true, label_prob, label_layer0_prob, label_layer1_prob, label_layer2_prob])
            else:
                print(predict_layer_0_index,predict_layer_1_index,predict_layer_2_index)
    test_result = sorted(test_result, key = lambda x: x[1])
    test_result = test_result[::-1]
    pr_x = []
    pr_y = []
    correct = 0
    p_4 = 0
    for i, item in enumerate(test_result):
        correct += item[0]
        pr_y.append(float(correct) / (i + 1))
        pr_x.append(float(correct) / conf.total_recall)
    auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
    for i in range(len(pr_x)): 
        if pr_x[i] >= 0.4:
            print("precision at relll@0.4")
            p_4 = pr_y[i]
            print(pr_x[i])
            print(pr_y[i])
            break     
    print("test auc_local: ", auc)
    print("p_4", p_4)
    return auc, pr_x, pr_y, test_result




conf = config.Config()
os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpu
conf.load_train_data()
conf.load_test_data()
tree = Tree(conf)
conf.global_num_classes = tree.n_class
base_model = PCNN_ATT(conf)
policy = Policy(conf, tree.n_class, base_model)
policy.cuda()

#policy_optimizer = torch.optim.Adam(policy.parameters(), lr=conf.policy_lr, weight_decay=conf.policy_weight_decay)
policy_optimizer = torch.optim.SGD(policy.parameters(), lr = conf.policy_lr, weight_decay = conf.policy_weight_decay)

for name,parameters in policy.named_parameters():
    print(name, parameters.size())


criterion = torch.nn.CrossEntropyLoss()
if conf.is_training :
    train()
else:
    test()






