def test_epoch_by_max(epoch):
    print('test by max %d local model starts'%(epoch))
    if not conf.is_training:
        policy.load_state_dict(torch.load("./checkpoint/epoch_lr0.5_" + str(epoch)))  
    conf.set_test_model(policy.base_model)
    policy.eval()
    test_result = []
    conf.acc_NA_local.clear()
    conf.acc_not_NA_local_layer0.clear()
    conf.acc_not_NA_local_layer1.clear()
    conf.acc_not_NA_local_layer2.clear()
    conf.acc_total_local.clear()
        #test flat model for comparation
    # conf.acc_NA_global.clear()
    # conf.acc_not_NA_global.clear()
    # conf.acc_total_global.clear()
    # conf.testModel = policy.base_model
    # auc, pr_x, pr_y = conf.test_one_epoch()
    # print("auc_flat:", auc)
    if conf.flat_probs_only:
        return

    bagid_label2prob_dict = defaultdict()
    predict_label2num = defaultdict(int)
    pred_not_na = 0
    correct_layer0 = 0
    correct_layer1 = 0
    correct_layer2 = 0
    for batch_num in tqdm(range(conf.test_batches)):
        conf.get_test_batch(batch_num)
        conf.test_one_step()
        logits = policy.base_model.test_hierarchical() #[160,96,690]
        #print("logits", logits, logits.size())
        policy.bag_vec_test = logits
        bag_ids = conf.bag_ids
        cur_batch_size = len(bag_ids)
        cur_class_batch = np.zeros(cur_batch_size, dtype=int)

        for layer in range(conf.n_layers):#3
            conf.cur_layer = layer
            next_classes_batch = tree.p2c_batch(cur_class_batch)#[batch,上一阶段标签的子标签]，可以看成第n层及他之前的标签
            indices, _ = np.where(next_classes_batch != 0)
            indices= np.array(list(set(indices)))
            if len(indices) == 0:
                break        
            next_classes_batch = next_classes_batch[indices]
            cur_class_batch = cur_class_batch[indices]
            bag_ids = [bag_ids[idx] for idx in indices]
            policy.get_test_bag_vec(next_classes_batch, indices) #根据next_classes_batch选择 self.bag_vec 
            h_probs = policy.step_sl_test(conf, cur_class_batch, next_classes_batch)

            #print("h_probs", h_probs, h_probs.size())
            next_class_batch_pred, next_class_batch_pred_prob = tree.get_next_by_probs(conf, cur_class_batch, next_classes_batch, bag_ids, h_probs, indices, cur_step = 0)
            cur_class_batch = next_class_batch_pred
            h_probs_np = h_probs.cpu().detach().numpy()

            #top 1
            for i, var in enumerate(indices):
                y_pred = next_class_batch_pred[i]
                y_true = tree.test_hierarchical_bag_multi_label[bag_ids[i]]
                cur_bag_id = bag_ids[i]
                bagid_label = str(cur_bag_id) + "_" + str(y_pred)
                if y_pred != 1:
                    bagid_label2prob_dict[bagid_label] = float(next_class_batch_pred_prob[i])
                    if layer == 0:
                        conf.acc_not_NA_local_layer0.add(y_pred in y_true)
                        ans = (y_pred in y_true)
                        correct_layer0 += ans
                    elif layer == 1:
                        conf.acc_not_NA_local_layer1.add(y_pred in y_true)
                        ans = (y_pred in y_true)
                        correct_layer1 += ans
                    elif layer == 2:
                        pred_not_na += 1
                        predict_label2num[y_pred] += 1
                        conf.acc_not_NA_local_layer2.add(y_pred in y_true)
                        ans = (y_pred in y_true)
                        test_result.append([ans,next_class_batch_pred_prob[i], y_pred, y_true])
                        correct_layer2 += ans
                else:
                    conf.acc_NA_local.add(y_pred in y_true)

    test_result = sorted(test_result, key = lambda x: x[1])
    test_result = test_result[::-1]
    pr_x = []
    pr_y = []
    correct = 0
    for i, item in enumerate(test_result):
        correct += item[0]
        pr_y.append(float(correct) / (i + 1))
        pr_x.append(float(correct) / conf.total_recall)
    auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
    for i in range(len(pr_x)): 
        if pr_x[i] >= 0.4:
            print("precision at relll@0.4")
            print(pr_x[i])
            print(pr_y[i])
            break
    print("test predict_label2num", predict_label2num, "pred_not_na", pred_not_na)
    print("correct_layer0", correct_layer0, "correct_layer1", correct_layer1, "correct_layer2", correct_layer2)
    print("test local model : NA accuracy: acc_NA_local: %f, layer0 accuracy: %f, layer1 accuracy: %f, layer2 accuracy: %f" % (conf.acc_NA_local.get(),conf.acc_not_NA_local_layer0.get(), conf.acc_not_NA_local_layer1.get(), conf.acc_not_NA_local_layer2.get()))       
    print("test auc_local_max_v1: ", auc)


    # test_result_all 
    test_result = []
    for bag_id in tqdm(range(len(tree.test_hierarchical_bag_multi_label))):
        y_true = tree.test_hierarchical_bag_multi_label[bag_id]
        for i in range(1, len(conf.test_batch_attention_query)):
            indices = conf.test_batch_attention_query[i]
            layer_0_index = str(bag_id) + "_" + str(indices[0])
            layer_1_index = str(bag_id) + "_" + str(indices[1])
            layer_2_index = str(bag_id) + "_" + str(indices[2])
            if layer_2_index in bagid_label2prob_dict:
                prob_i = bagid_label2prob_dict[layer_0_index] * bagid_label2prob_dict[layer_1_index] * bagid_label2prob_dict[layer_2_index]
                ans = int(indices[2] in y_true)
                test_result.append([ans, prob_i, indices[2], y_true])
    test_result = sorted(test_result, key = lambda x: x[1])
    test_result = test_result[::-1]
    pr_x = []
    pr_y = []
    correct = 0
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
    print("test auc_local_max_v2: ", auc)
    file_name = "./test_result/bagid_label2prob_max_v2_epoch" + str(epoch) + ".json"
    with open(file_name, "w") as file:
        json.dump(bagid_label2prob_dict, file)
    return auc, pr_x, pr_y, p_4

    #在test_result_all  最下面
     # test_result_all 
    test_result = []
    for bag_id in tqdm(range(len(tree.test_hierarchical_bag_multi_label))):
        y_true = tree.test_hierarchical_bag_multi_label[bag_id]
        for i in range(1, len(conf.test_batch_attention_query)):
            indices = conf.test_batch_attention_query[i]
            layer_0_index = str(bag_id) + "_" + str(indices[0])
            layer_1_index = str(bag_id) + "_" + str(indices[1])
            layer_2_index = str(bag_id) + "_" + str(indices[2])
            prob_i = bagid_label2prob_dict[layer_0_index] * bagid_label2prob_dict[layer_1_index] * bagid_label2prob_dict[layer_2_index]
            ans = int(indices[2] in y_true)
            test_result.append([ans, prob_i, indices[2], y_true])
    test_result = sorted(test_result, key = lambda x: x[1])
    test_result = test_result[::-1]
    pr_x = []
    pr_y = []
    correct = 0
    for i, item in enumerate(test_result):
        correct += item[0]
        pr_y.append(float(correct) / (i + 1))
        pr_x.append(float(correct) / conf.total_recall)
    auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
    for i in range(len(pr_x)): 
        if pr_x[i] >= 0.4:
            print("precision at relll@0.4")
            print(pr_x[i])
            print(pr_y[i])
            break     
    print("test auc_local_all: ", auc)
    return auc, pr_x, pr_y   





indices = attention_query #[241,3]
print("indices", indices.size())
dummy = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), a_6.size(2))
print("dummy_1", dummy.size())
dummy = dummy.unsqueeze(3).expand(indices.size(0), indices.size(1), a_6.size(2),a_6.size(3))  #[241,3,120,300]
print("dummy_2", dummy.size())
a_7 = torch.gather(a_6, 1, dummy) #[241,3,120,300]
print("a_7", a_7.size())
word_repre = x_2.unsqueeze(1) #[241,1,120,300]
#print("word_repre", word_repre.size())

word_repre_1 = word_repre.expand(-1, attention_query.size(1), -1, -1) #[241,3,120,300]
print("word_repre_1", word_repre_1.size())

word_repre_2 = word_repre_1.mul(a_7) #[241,3,120,300] 
print("word_repre_2", word_repre_2.size())

word_repre_3 = torch.transpose(word_repre_2, 2, 3) #[241,3,300,120]
print("word_repre_3", word_repre_3.size())

word_repre_final = word_repre_3.unsqueeze(4)# #[241,3,300,120,1]
print("word_repre_final", word_repre_final.size())  






        gpu_tracker.track() 
        del a_5, a_4, a_3, a_2, a_1, x
        gpu_tracker.track() 
        word_repre_stack = []
        for i in range(self.config.global_num_classes):
            a_7 = torch.index_select(a_6, 1, self.class_tensor[i])
            print("a_7", a_7.size())
            word_repre = x_2.unsqueeze(1) #[241,1,120,300]
            word_repre_label = word_repre.mul(a_7) #[241,1,120,300]
            word_repre_stack.append(word_repre_label) 
            print("word_repre_stack", len(word_repre_stack))
            gpu_tracker.track() 

        word_repre = torch.stack(word_repre_stack,1) #[241,95,120,300]
        print("len(word_repre)", len(word_repre))

        exit()



    # def test(self, x):
    #   attention_logit = self._attention_test_logit(x) #[all_sen_num, 53]
    #   tower_output = []
    #   for i in range(len(self.scope) - 1):
    #       sen_matrix = x[self.scope[i] : self.scope[i + 1]]
    #       attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i] : self.scope[i + 1]], 0, 1), 1) #[53,bag_sen_num]
    #       final_repre = torch.matmul(attention_score, sen_matrix) #[53, 690] = [53,bag_sen_num] * [bag_sen_num, 690]
    #       probs = self.get_logits(final_repre) #[53,53]
    #       tower_output.append(torch.diag(F.softmax(probs, 1)))#[1,53]
    #   stack_output = torch.stack(tower_output)
    #   return list(stack_output.data.cpu().numpy())





    #from torch.autograd import Variable
# from gpu_mem_track import MemTracker
# import inspect
# frame = inspect.currentframe()     
# gpu_tracker = MemTracker(frame)  
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
class Label_Aware_Word_Att(nn.Module):
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


class Word_ATT(nn.Module):
    def __init__(self, config):
        super(Word_ATT, self).__init__()
        self.config = config
        self.w1= nn.Embedding(60, 60) #[60,60]
        self.w2 = nn.Embedding(60, 60)#[60,60]
        self.activation = nn.Tanh()
        nn.init.xavier_uniform_(self.w1.weight.data)
        nn.init.xavier_uniform_(self.w2.weight.data)
        torch.set_printoptions(precision=8)
    def forward(self, x): #x=[241,120,60]
        # x_1 = torch.transpose(x, 1, 2) #[241,60,120]
        a_1 = torch.matmul(x, self.w1.weight)  #[241,120,60] =  [241, 120, 60] * [60,60]  
        # print("a_1", a_1.size())
        a_1 = self.activation(a_1)
        a_2 = torch.matmul(a_1, self.w2.weight) #[241,120,60] = [241,120,60] * [60,60]
        # print("a_2", a_2.size())
        a_4 = F.softmax(a_2,1) #[241,120,60]
        print("a_4", a_4[0], a_4.size())
        # a_4 = a_4.unsqueeze(2)
        # a_5 = a_4.expand(-1, 120, 60) #[241,120, 60]
        # print("a_5", a_5[0], a_5.size())
        x_final = x.mul(a_4) #[241,120,60] = [241,120,60] * [241,120,60]
        #print("x_final", x_final.size())
        return x_final
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
        attention_logit = self._attention_test_logit(x) #[all_sen_num, 95, 95]
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
        attention_logit = self._attention_test_logit(x) #[all_sen_num, 95]
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i] : self.scope[i + 1]] #[bag_sen_num, 690]
            attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i] : self.scope[i + 1]], 0, 1), 1) #[95,bag_sen_num]
            logits = torch.matmul(attention_score, sen_matrix) #[95, 690] = [95,bag_sen_num] * [bag_sen_num, 690]
            tower_repre.append(logits)
        stack_repre = torch.stack(tower_repre)#[160,95,690]
        return stack_repre
