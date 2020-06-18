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
