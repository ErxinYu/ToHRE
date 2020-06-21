import numpy as np
import os
import json
from collections import defaultdict, Counter

in_path = "./raw_data/57K/"
out_path = "./data/57K-combined-unjoined-first/"
#out_path = "./data"
case_sensitive = False
if not os.path.exists(out_path):
        os.mkdir(out_path)
train_file_name = in_path + 'train_combined_unjoined_first.json'
test_file_name = in_path + 'test_combined_unjoined_first.json'
word_file_name = in_path + 'word_vec.json'
rel_file_name = in_path + 'rel_desc2id.json'


def gen_hierarchical_label():
        h_relation2id = {}

        p2c = defaultdict(list)
        p2c["root"] = ["NA"]
        p2c["NA"] = []
        layer1_label_set = set()
        layer2_label_set = set()
        layer3_label_set = set()
        all_layer_label2num = defaultdict(int)
        with open(rel_file_name, 'r') as file:
                rel2id_origin = json.load(file)
        print(rel2id_origin)
        rel2id = sorted(rel2id_origin.items(), key=lambda item:item[1])
        print(rel2id)

        #for label in rel2id_origin:
        for label_num in rel2id:
                label = label_num[0]
                if label != "NA":
                        layer1_label = "/" + label.split("/")[1]
                        layer2_label = layer1_label + "/" + label.split("/")[2]
                        layer3_label = layer2_label + "/" + label.split("/")[3]

                        layer1_label_set.add(layer1_label) 
                        layer2_label_set.add(layer2_label)
                        layer3_label_set.add(layer3_label)

                        p2c["root"].append(layer1_label)
                        p2c[layer1_label].append(layer2_label)
                        p2c[layer2_label].append(layer3_label)
                        p2c[layer3_label] = []
        for parent in p2c:
                p2c[parent] = list(set(p2c[parent]))

        #计算h_relation2id
        h_relation2id["root"] = 0
        h_relation2id["NA"] = 1
        id = 2
        layer1_label_list = sorted(list(layer1_label_set))
        layer2_label_list = sorted(list(layer2_label_set))
        layer3_label_list = sorted(list(layer3_label_set))
        for label in layer1_label_list:
                h_relation2id[label] = id
                id += 1
        for label in layer2_label_list:
                h_relation2id[label] = id
                id += 1
        for label in layer3_label_list:
                h_relation2id[label] = id
                id += 1
        #计算p2c_id
        p2c_id = defaultdict(list)
        for parent in p2c:
                if len(p2c[parent]) != 0:
                        for rel in p2c[parent]:
                                p2c_id[h_relation2id[parent]].append(h_relation2id[rel])
                else:
                        p2c_id[h_relation2id[parent]] = []
        print("h_relation2id", h_relation2id, len(h_relation2id))
        #计算relation2h_relation_id
        relation_id2h_relation_id = defaultdict(list)
        for label in rel2id_origin:
                if label != "NA":
                        layer1_label = "/" + label.split("/")[1]
                        layer2_label = layer1_label + "/" + label.split("/")[2]
                        layer3_label = layer2_label + "/" + label.split("/")[3] 
                        relation_id2h_relation_id[rel2id_origin[label]].append(h_relation2id[layer1_label])
                        relation_id2h_relation_id[rel2id_origin[label]].append(h_relation2id[layer2_label])
                        relation_id2h_relation_id[rel2id_origin[label]].append(h_relation2id[layer3_label])
        for parent in relation_id2h_relation_id:
                relation_id2h_relation_id[parent] = list(set(relation_id2h_relation_id[parent]))
                relation_id2h_relation_id[parent].sort()
        relation_id2h_relation_id[0] = [1,1,1]
        with open(out_path + "relation_id2h_relation_id.json", "w") as file:
                json.dump(relation_id2h_relation_id, file)

        with open(out_path + "p2c_id.json", "w") as file:
                json.dump(p2c_id, file)

        return relation_id2h_relation_id

def find_pos(sentence, head, tail):

        def find(sentence, entity):
                p = sentence.find(' ' + entity + ' ')
                if p == -1:
                        if sentence[:len(entity) + 1] == entity + ' ':
                                p = 0
                        elif sentence[-len(entity) - 1:] == ' ' + entity:
                                p = len(sentence) - len(entity)
                        else:
                                p = 0
                else:
                        p += 1
                return p
        sentence = ' '.join(sentence.split())   
        p1 = find(sentence, head)
        p2 = find(sentence, tail)
        words = sentence.split()
        cur_pos = 0 
        pos1 = -1
        pos2 = -1
        for i, word in enumerate(words):
                if cur_pos == p1:
                        pos1 = i
                if cur_pos == p2:
                        pos2 = i
                cur_pos += len(word) + 1
        # print(head)
        # print(tail)
        # print(sentence)
        # print(pos1, pos2)
        # print("/n")

        return pos1, pos2
                
def init(file_name, word_vec_file_name, rel2id_file_name, max_length = 120, case_sensitive = False, is_training = True):
        if file_name is None or not os.path.isfile(file_name):
                raise Exception("[ERROR] Data file doesn't exist")
        if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
                raise Exception("[ERROR] Word vector file doesn't exist")
        if rel2id_file_name is None or not os.path.isfile(rel2id_file_name):
                raise Exception("[ERROR] rel2id file doesn't exist")
        
        relation_id2h_relation_id = gen_hierarchical_label()
        print("Loading data file...")
        ori_data = json.load(open(file_name, "r"))
        print("Finish loading")
        print("Loading word_vec file...")
        ori_word_vec = json.load(open(word_vec_file_name, "r"))
        print("Finish loading")
        print("Loading rel2id file...")
        rel2id = json.load(open(rel2id_file_name, "r"))
        print("Finish loading")
        
        if not case_sensitive:
                print("Eliminating case sensitive problem...")
                for i in ori_data:
                        i['sentence'] = i['sentence'].lower()
                        i['head']['word'] = i['head']['word'].lower()
                        i['tail']['word'] = i['tail']['word'].lower()
                for i in ori_word_vec:
                        i['word'] = i['word'].lower()
                print("Finish eliminating")
        
        # vec
        print("Building word vector matrix and mapping...")
        word2id = {}
        word_vec_mat = []
        word_size = len(ori_word_vec[0]['vec'])
        print("Got {} words of {} dims".format(len(ori_word_vec), word_size))
        for i in ori_word_vec:
                word2id[i['word']] = len(word2id)
                word_vec_mat.append(i['vec'])
        word2id['UNK'] = len(word2id)
        word2id['BLANK'] = len(word2id)
        word_vec_mat.append(np.random.normal(loc = 0, scale = 0.05, size = word_size))
        word_vec_mat.append(np.zeros(word_size, dtype = np.float32))
        word_vec_mat = np.array(word_vec_mat, dtype = np.float32)
        print("Finish building")
        
        # sorting
        print("Sorting data...")
        ori_data.sort(key = lambda a: a['head']['id'] + '#' + a['tail']['id'] + '#' + a['relation'])
        print("Finish sorting")
        
        sen_tot = len(ori_data)
        h_entity_word = np.zeros((sen_tot, 1), dtype = np.int64)
        t_entity_word = np.zeros((sen_tot, 1), dtype = np.int64)
        sen_word = np.zeros((sen_tot, max_length), dtype = np.int64)
        sen_pos1 = np.zeros((sen_tot, max_length), dtype = np.int64)
        sen_pos2 = np.zeros((sen_tot, max_length), dtype = np.int64)
        sen_mask = np.zeros((sen_tot, max_length, 3), dtype = np.float32)
        sen_label = np.zeros((sen_tot), dtype = np.int64)
        sen_len = np.zeros((sen_tot), dtype = np.int64)
        bag_label = []
        hierarchical_bag_label = []
        hierarchical_ins_label = []
        bag_key_dict = defaultdict(list)
        bag_scope = []
        bag_key = []
        relation_out_range = 0
        for i in range(len(ori_data)):
                if  i%10000 == 0:
                        print(i)

                sen = ori_data[i]
                # sen_label 
                #rule 1
                if sen['relation'] == "/base/locations/countries/states_provinces_within":
                        sen['relation'] = "/location/country/states_provinces_withins"
                
                if sen['relation'] in rel2id:
                        sen_label[i] = rel2id[sen['relation']]
                else:
                        relation_out_range += 1
                        print("relation not in 53", sen['relation'], relation_out_range)
                        sen_label[i] = rel2id['NA']
                words = sen['sentence'].split()
                # sen_len
                sen_len[i] = min(len(words), max_length)
                # sen_word
                for j, word in enumerate(words):
                        if j < max_length:
                                if word in word2id:
                                        sen_word[i][j] = word2id[word]
                                else:
                                        sen_word[i][j] = word2id['UNK']
                for j in range(j + 1, max_length):
                        sen_word[i][j] = word2id['BLANK']

                pos1, pos2 = find_pos(sen['sentence'], sen['head']['word'], sen['tail']['word'])
                if pos1 == -1 or pos2 == -1:
                        raise Exception("[ERROR] Position error, index = {}, sentence = {}, head = {}, tail = {}".format(i, sen['sentence'], sen['head']['word'], sen['tail']['word']))
                if pos1 >= max_length:
                        pos1 = max_length - 1
                if pos2 >= max_length:
                        pos2 = max_length - 1
                pos_min = min(pos1, pos2)
                pos_max = max(pos1, pos2)
                for j in range(max_length):
                        # sen_pos1, sen_pos2
                        sen_pos1[i][j] = j - pos1 + max_length
                        sen_pos2[i][j] = j - pos2 + max_length
                        # sen_mask
                        if j >= sen_len[i]:
                                sen_mask[i][j] = [0, 0, 0]
                                sen_pos1[i][j] = max_length
                                sen_pos1[i][j] = max_length
                        elif j - pos_min <= 0:
                                sen_mask[i][j] = [100, 0, 0]
                        elif j - pos_max <= 0:
                                sen_mask[i][j] = [0, 100, 0]
                        else:
                                sen_mask[i][j] = [0, 0, 100]    
                # bag_scope     
                tup = (sen['head']['id'], sen['tail']['id'], sen['relation'])

                if bag_key == [] or bag_key[len(bag_key) - 1] != tup:
                        bag_key.append(tup)
                        bag_scope.append([i, i])
                bag_scope[len(bag_scope) - 1][1] = i

        print("Processing bag label...")
        print("bag_key", len(bag_key), bag_key[0])
        for index, i in enumerate(bag_scope):
                bag_label.append(sen_label[i[0]])
                hierarchical_bag_label.append(relation_id2h_relation_id[sen_label[i[0]]])
                tup = (bag_key[index][0], bag_key[index][1], int(sen_label[i[0]]))
                bag_key_dict[index] = tup

        print("Finish processing")
        # ins_scope
        ins_scope = np.stack([list(range(len(ori_data))), list(range(len(ori_data)))], axis = 1)
        print("Processing instance label...")
        
        # ins_label
        ins_label = sen_label
        for sen_id in range(len(ins_label)):
                hierarchical_ins_label.append(relation_id2h_relation_id[ins_label[sen_id]])


        print("Finishing processing")


        bag_scope = np.array(bag_scope, dtype = np.int64)
        bag_label = np.array(bag_label, dtype = np.int64)
        ins_scope = np.array(ins_scope, dtype = np.int64)
        ins_label = np.array(ins_label, dtype = np.int64)
        hierarchical_bag_label = np.array(hierarchical_bag_label, dtype = np.int64)
        hierarchical_ins_label = np.array(hierarchical_ins_label, dtype = np.int64)
        # saving
        print("Saving files")
        if is_training:
                name_prefix = "train"
        else:
                name_prefix = "test"
        np.save(os.path.join(out_path, 'vec.npy'), word_vec_mat)
        np.save(os.path.join(out_path, name_prefix + '_word.npy'), sen_word)
        np.save(os.path.join(out_path, name_prefix + '_pos1.npy'), sen_pos1)
        np.save(os.path.join(out_path, name_prefix + '_pos2.npy'), sen_pos2)
        np.save(os.path.join(out_path, name_prefix + '_mask.npy'), sen_mask)
        np.save(os.path.join(out_path, name_prefix + '_bag_label.npy'), bag_label)
        np.save(os.path.join(out_path, name_prefix + '_bag_scope.npy'), bag_scope)
        np.save(os.path.join(out_path, name_prefix + '_ins_label.npy'), ins_label)
        np.save(os.path.join(out_path, name_prefix + '_ins_scope.npy'), ins_scope)
        np.save(os.path.join(out_path, name_prefix + '_hierarchical_bag_label.npy'), hierarchical_bag_label)
        np.save(os.path.join(out_path, name_prefix + '_hierarchical_ins_label.npy'), hierarchical_ins_label)
        with open(os.path.join(out_path, name_prefix + '_bag_key.json'),"w") as file:
                json.dump(bag_key_dict, file)
        print("Finish saving")          

init(train_file_name, word_file_name, rel_file_name, max_length = 120, case_sensitive = False, is_training = True)
init(test_file_name, word_file_name, rel_file_name, max_length = 120, case_sensitive = False, is_training = False)
gen_hierarchical_label()
