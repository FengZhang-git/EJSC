import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import time
from sklearn import metrics


from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def intent_prototype_loss(model_outputs, num_classes, num_support, num_query):

    outputs = model_outputs.mean(dim=1)
    support_size = int(num_support.sum())
    intent_support_tensor = outputs[:support_size][:]
    intent_query_tensor = outputs[support_size:][:]
    start = 0
    support_prototype = None
    for i, num_support_per_class in enumerate(num_support):
        support_tensors_per_intent = intent_support_tensor[start: start+num_support_per_class]
        if support_prototype is None:
            support_prototype = support_tensors_per_intent.mean(dim=0)
        else:
            support_prototype = torch.cat([support_prototype, support_tensors_per_intent.mean(dim=0)], 0)

        start = start + num_support_per_class
    support_prototype = support_prototype.view(num_classes, model_outputs.shape[2])
    dists = euclidean_dist(intent_query_tensor, support_prototype)
    

    log_p_y = F.log_softmax(-dists, dim=1).view(num_classes, num_query, -1)

    target_inds = torch.arange(0, num_classes)
    target_inds = target_inds.view(num_classes, 1, 1)
    target_inds = target_inds.expand(num_classes, num_query, 1).long()
    target_inds = target_inds.cuda()

    loss_pn = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_pn = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_pn, acc_pn, y_hat


def intent_scl_loss(model_outputs, num_classes, num_support, num_query, temperature=0.07):
    
    outputs = model_outputs.mean(dim=1)
    outputs = F.normalize(outputs, p=2, dim=1)

    support_size = int(num_support.sum())
    # k_s X 768
    intent_support_tensor = outputs[:support_size][:]
    intent_query_tensor = outputs[support_size:][:]

    anchor_dot_contrast = torch.div(
            torch.matmul(intent_query_tensor, intent_support_tensor.T),
            temperature)
    # for numerical stability, 
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    mask = torch.zeros(num_classes*num_query, num_support.sum())
    mask = mask.cuda()
    start = 0
    for i, num in enumerate(num_support):
        tmp = torch.zeros(1, num_support.sum())
        tmp[0][start:start+num] = 1
        mask[i*num_query:(i+1)*num_query] = tmp
        start = start + num
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    if torch.any(torch.isnan(log_prob)):
        raise ValueError("Log_prob has nan!")
    
    
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1))

    loss = - mean_log_prob_pos.mean()

    return loss




def get_slot_information(model_outputs, num_classes, num_support, num_query, slots, window=2):

    
    support_size = int(num_support.sum())
    slot_support_tensor = model_outputs[:support_size][:][:]
    slot_query_tensor = model_outputs[support_size:][:][:]
    slot_support_labels = slots[:support_size]
    slot_query_labels = slots[support_size:]
  
    query_slot_set = []
    for slots_per_example in slot_query_labels:   
        for each_slot in slots_per_example:
            if each_slot not in query_slot_set:
                query_slot_set.append(each_slot)

   
    slot_dict = defaultdict(list)
    for i in range(support_size):
        slots_per_example = slot_support_labels[i]
        seq_len = len(slots_per_example)
        for index, each_slot in enumerate(slots_per_example):
            
          
            if each_slot == 'F' or each_slot not in query_slot_set:
                continue
            left = (index+1-window) if (index+1-window) > 1 else 1
            right = (index+1+window) if (index+window) < seq_len else seq_len
            slot_tensor = slot_support_tensor[i][left:right+1][:]
            slot_tensor = slot_tensor.mean(dim=0).unsqueeze(0)
            slot_dict[each_slot].append(slot_tensor)
    slot_to_ids = defaultdict(int)
    ids_to_slot = defaultdict(str)

    
    for i, key in enumerate(slot_dict.keys()):
        slot_to_ids[key] = i
        ids_to_slot[i] = key

       
    slot_query_labels_ids = []
    seq_lens = []
    slot_query_tensor_list = []
    for i, slots_per_query in enumerate(slot_query_labels):
        k = 0
        for j, each_query_slot in enumerate(slots_per_query):
            if each_query_slot not in slot_dict.keys():
                k += 1
                continue
            slot_query_labels_ids.append(slot_to_ids[each_query_slot])
            slot_query_tensor_list.append(slot_query_tensor[i][j+1][:])

        seq_lens.append(len(slots_per_query)-k)
        
    
    slot_query_tensor_cat = torch.stack(slot_query_tensor_list)

    return slot_dict, slot_query_labels_ids, slot_query_tensor_cat, ids_to_slot, seq_lens


def slot_prototype_loss(model_outputs, num_classes, num_support, num_query, slots, window=2):
    

    slot_dict, slot_query_labels_ids, slot_query_tensor_cat, ids_to_slot, seq_lens = get_slot_information(model_outputs, num_classes, num_support, num_query, slots, window)

    slot_prototypes = []
    for i, key in enumerate(slot_dict.keys()):
        
        slot_prototype_per_class = torch.stack(slot_dict[key]).mean(0)
        slot_prototypes.append(slot_prototype_per_class)
    
    
    matrix_slot_prototypes = torch.stack(slot_prototypes).squeeze(1)


    if len(slot_query_labels_ids) != slot_query_tensor_cat.shape[0]:
        raise ValueError("The number of labels of slots is wrong.")

    dists = euclidean_dist(slot_query_tensor_cat, matrix_slot_prototypes)
    
    log_p_y = F.log_softmax(-dists, dim=1)
    target_inds = torch.tensor(slot_query_labels_ids).unsqueeze(1)
    target_inds = target_inds.cuda()
    loss_pn = -log_p_y.gather(1, target_inds).squeeze().view(-1).mean()

    _, y_hat = log_p_y.max(1)
    
    target_inds = target_inds.squeeze()
    y_true = []
    y_pred = []

    start = 0
    for seq_len in seq_lens:
        
        target_slots = []
        predict_slots = []
        for i in range(seq_len):
            target_slots.append(ids_to_slot[target_inds[start+i].item()])
            predict_slots.append(ids_to_slot[y_hat[start+i].item()])
        y_true.append(target_slots)
        y_pred.append(predict_slots)
        start = start + seq_len    


     
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    

    return loss_pn, acc, f1

    

def slot_scl_loss(model_outputs, num_classes, num_support, num_query, slots, window=2, temperature=0.07):

    slot_dict, slot_query_labels_ids, slot_query_tensor_cat, _, _ = get_slot_information(model_outputs, num_classes, num_support, num_query, slots, window)

    total_support_list, num_support_slot = [], []
    support_ids_interval = defaultdict(list)

    start = 0
    for i, key in enumerate(slot_dict.keys()):
        total_support_list.extend(slot_dict[key])
        num_support_slot.append(len(slot_dict[key]))
        support_ids_interval[i] = [start, start+len(slot_dict[key])]
        start = start + len(slot_dict[key])
    
    total_support_tensor = torch.stack(total_support_list).squeeze(dim=1)
    
    # normalize the features
    total_support_tensor = F.normalize(total_support_tensor, p=2, dim=1)
    slot_query_tensor_cat = F.normalize(slot_query_tensor_cat, p=2, dim=1)
    

    anchor_dot_contrast = torch.div(
            torch.matmul(slot_query_tensor_cat, total_support_tensor.T),
            temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    mask = torch.zeros(logits.shape[0], logits.shape[1])
    mask = mask.cuda()
    
    for i, ids in enumerate(slot_query_labels_ids):
        left, right = support_ids_interval[ids]
        mask[i][left:right] = 1

    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    if torch.any(torch.isnan(log_prob)):
        raise ValueError("Log_prob has nan!")
    
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1))

    loss_scl = - mean_log_prob_pos.mean()

    return loss_scl

    


    


class MixedLoss(nn.Module):
    def __init__(self, mode=None, window=1, temperature=0.07, lamda1=0.1, lamda2=0.1, lamda3=0.1):
        super(MixedLoss, self).__init__()
        self.window = window
        self.temperature = temperature
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.lamda3 = lamda3
        print(f"Lamda1:{lamda1}, Lamda2:{lamda2}, Lamda3:{lamda3}, Window:{window}")
        
    

    def forward(self, model_outputs, num_classes, class_ids, num_support, num_query, support_labels, query_labels, slots):
        
        word_tensor, intent_outputs, slot_outputs = model_outputs
        num_support = torch.tensor(num_support)

        intent_loss_pn, intent_acc_pn, y_hat = intent_prototype_loss(intent_outputs, num_classes, num_support, num_query)
        if self.lamda2 != 0:
            intent_loss_scl = intent_scl_loss(word_tensor, num_classes, num_support, num_query, self.temperature)
        else:
            intent_loss_scl = 0
        slot_loss_pn, slot_acc_pn, slot_f1_pn = slot_prototype_loss(slot_outputs, num_classes, num_support, num_query, slots, self.window)
        if self.lamda3 != 0:
            slot_loss_scl = slot_scl_loss(word_tensor, num_classes, num_support, num_query, slots, self.window, self.temperature)
        else:
            slot_loss_scl = 0
        acc = intent_acc_pn 

        if torch.isnan(intent_loss_pn):
            print("Intent loss is nan!")
        if torch.isnan(slot_loss_pn):
            print("Slot loss is nan!")
        loss = intent_loss_pn + self.lamda1 * slot_loss_pn + self.lamda2 * intent_loss_scl + self.lamda3 * slot_loss_scl 

        if False:
            return (loss, acc, slot_f1_pn, y_hat)


        return (loss, acc, slot_f1_pn)
       






