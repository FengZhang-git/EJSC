

import os
import torch
import numpy as np
from tqdm import tqdm
from parser_util import get_parser
from data_process.data_loader import Dataloader
from encoder.encoder import ModelManager
from transformers import AdamW, get_linear_schedule_with_warmup


from losses import MixedLoss
import json

def write_predicts(query_text, query_labels, num_classes, num_query, y_hat, mode='train'):
    y_hat = y_hat.view(num_classes*num_query)
    predicts = []
    for i in y_hat:
        ind = int(i)
        predicts.append(query_labels[ind*num_query])
    with open("atis/test1/test.json", "a+") as fout:
        for i, line in enumerate(query_text):
            tmp = {"text_u": line, "intent": query_labels[i], "predicts": predicts[i]}
            fout.write("%s\n" % json.dumps(tmp, ensure_ascii=False))



def init_dataloader(args, mode):
    filePath = os.path.join(args.dataFile, mode + '.json')
    vocabPath = os.path.join(args.vocabFile, mode +  '.intent')
    dataloader = Dataloader(filePath=filePath, vocabPath=vocabPath)

    return dataloader


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def init_model(args):
    device = torch.device('cuda', args.numDevice)
    torch.cuda.set_device(device)
    model = ModelManager(args).to(device)
    return model

def init_optim(args, model):

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    


    return optimizer

def init_lr_scheduler(args, optim):
    '''
    Initialize the learning rate scheduler
    '''
    t_total = args.epochs * args.num_episode_per_epoch
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    return scheduler


def train(args, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    device = torch.device('cuda', args.numDevice)

    if val_dataloader is None:
        # best_state = None
        acc_best_state = None
        f1_best_state = None 
    train_loss, epoch_train_loss = [], []
    train_acc, epoch_train_acc = [], []
    train_f1, epoch_train_f1 = [], []
    val_loss, epoch_val_loss = [], []
    val_acc, epoch_val_acc = [], []
    val_f1, epoch_val_f1 = [], []
    best_acc = 0
    best_f1 = 0
    loss_fn = MixedLoss(mode='mixed', window = 0, temperature=args.temperature, lamda1=args.lamda1, lamda2=args.lamda2, lamda3=args.lamda3)
    
    # best_model_path = os.path.join(args.fileModelSave, 'best_model.pth')
    acc_best_model_path = os.path.join(args.fileModelSave, 'acc_best_model.pth')
    f1_best_model_path = os.path.join(args.fileModelSave, 'f1_best_model.pth')

    last_model_path = os.path.join(args.fileModelSave, 'last_model.pth')
    for epoch in range(args.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        model.train()
        
        for episode in tqdm(range(args.num_episode_per_epoch)):
            optim.zero_grad()
            datas = tr_dataloader.get_episode_datas()
            num_classes, class_ids, num_support, num_query, episode_datas = datas
            support_labels, support_text, query_labels, query_text, slots = tr_dataloader.split_data(episode_datas)
            
            intent_description, slots_description = [], []
            for label in query_labels:
                if label not in intent_description:
                    intent_description.append(label)

            for slot_per_instance in slots:
                for slot in slot_per_instance:
                    if slot == 'F':
                        continue
                    if len(slot) >= 2:
                        slot = slot[2:]
                    if slot not in slots_description:
                        slots_description.append(slot)

            text_list = support_text + query_text




            model_outputs = model(text_list, intent_description, slots_description)

            loss, acc, f1 = loss_fn(model_outputs, 
                                num_classes=num_classes, 
                                class_ids=class_ids,
                                num_support=num_support,
                                num_query=num_query,
                                support_labels=support_labels,
                                query_labels=query_labels,
                                slots=slots)
            


            loss.backward()
            optim.step()
            lr_scheduler.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
            train_f1.append(f1)
            # print(f"Episode: {episode}: Loss: {loss.item()}, Acc: {acc.item()}, F1: {f1}")
            
        avg_loss = np.mean(train_loss[-args.num_episode_per_epoch:])
        avg_acc = np.mean(train_acc[-args.num_episode_per_epoch:])
        avg_f1 = np.mean(train_f1[-args.num_episode_per_epoch:])
        print('Avg Train Loss: {}, Avg Train Acc: {}, Avg Train F1: {}'.format(avg_loss, avg_acc, avg_f1))
        epoch_train_loss.append(avg_loss)
        epoch_train_acc.append(avg_acc)
        epoch_train_f1.append(avg_f1)

        if val_dataloader is None:
            continue
        
        model.eval()
        for step in tqdm(range(args.num_episode_val)):

            datas = tr_dataloader.get_episode_datas()
            num_classes, class_ids, num_support, num_query, episode_datas = datas
            support_labels, support_text, query_labels, query_text, slots = tr_dataloader.split_data(episode_datas)
            text_list = support_text + query_text

            intent_description, slots_description = [], []
            for label in query_labels:
                if label not in intent_description:
                    intent_description.append(label)

            for slot_per_instance in slots:
                for slot in slot_per_instance:
                    if slot == 'F':
                        continue
                    if len(slot) >= 2:
                        slot = slot[2:]
                    if slot not in slots_description:
                        slots_description.append(slot)

            model_outputs = model(text_list, intent_description, slots_description)

            loss, acc, f1 = loss_fn(model_outputs, 
                                num_classes=num_classes, 
                                class_ids=class_ids,
                                num_support=num_support,
                                num_query=num_query,
                                support_labels=support_labels,
                                query_labels=query_labels,
                                slots=slots)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
            val_f1.append(f1)
            # print(f"Episode: {episode}: Loss: {loss.item()}, Acc: {acc.item()}, F1: {f1}")
        avg_loss = np.mean(val_loss[-args.num_episode_val:])
        avg_acc = np.mean(val_acc[-args.num_episode_val:])
        avg_f1 = np.mean(val_f1[-args.num_episode_val:])

        epoch_val_loss.append(avg_loss)
        epoch_val_acc.append(avg_acc)
        epoch_val_f1.append(avg_f1)

        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        f_postfix = ' (Best)' if avg_f1 >= best_f1 else ' (Best: {})'.format(
            best_f1)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}, Avg Val F1: {}{}'.format(
            avg_loss, avg_acc, postfix, avg_f1, f_postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), acc_best_model_path)
            best_acc = avg_acc
            acc_best_state = model.state_dict()

        if avg_f1 >= best_f1:
            torch.save(model.state_dict(), f1_best_model_path)
            best_f1 = avg_f1
            f1_best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    for name in ['epoch_train_loss', 'epoch_train_acc', 'epoch_train_f1', 'epoch_val_loss', 'epoch_val_acc', 'epoch_val_f1']:
        save_list_to_file(os.path.join(args.fileModelSave,
                                       name + '.txt'), locals()[name])

    return acc_best_state, f1_best_state, best_acc, train_loss, train_acc, val_loss, val_acc
        

def test(args, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = torch.device('cuda', args.numDevice)
    val_acc = []
    val_loss = []
    val_f1 = []
    loss_fn = MixedLoss(mode='mixed', window=0, temperature=args.temperature, lamda1=args.lamda1, lamda2=args.lamda2, lamda3=args.lamda3)
    model.eval()
    
    for batch in tqdm(range(args.num_episode_test)):


        datas = test_dataloader.get_episode_datas()
        num_classes, class_ids, num_support, num_query, episode_datas = datas
        support_labels, support_text, query_labels, query_text, slots = test_dataloader.split_data(episode_datas)
        text_list = support_text + query_text
        
        intent_description, slots_description = [], []
        for label in query_labels:
            if label not in intent_description:
                intent_description.append(label)

        for slot_per_instance in slots:
            for slot in slot_per_instance:
                if slot == 'F':
                    continue
                if len(slot) >= 2:
                    slot = slot[2:]
                if slot not in slots_description:
                    slots_description.append(slot)

        model_outputs = model(text_list, intent_description, slots_description)


        loss, acc, f1 = loss_fn(model_outputs, 
                            num_classes=num_classes, 
                            class_ids=class_ids,
                            num_support=num_support,
                            num_query=num_query,
                            support_labels=support_labels,
                            query_labels=query_labels,
                            slots=slots)
        
        # write_predicts(query_text, query_labels, num_classes, num_query, y_hat, 'test')

        val_loss.append(loss.item())
        val_acc.append(acc.item())
        val_f1.append(f1)
            

    avg_acc = np.mean(val_acc)
    avg_loss = np.mean(val_loss)
    avg_f1 = np.mean(val_f1)
    print('Test Acc: {}'.format(avg_acc))
    print('Test Loss: {}'.format(avg_loss))
    print('Test F1: {}'.format(avg_f1))
    path = args.fileModelSave + "/test_score.json"
    with open(path, "a+") as fout:
        tmp = {"Acc": avg_acc, "F1": avg_f1, "Loss": avg_loss}
        fout.write("%s\n" % json.dumps(tmp, ensure_ascii=False))

    return avg_acc, avg_loss, avg_f1


def write_args_to_josn(args):
    path = args.fileModelSave + "/config.json"
    args = vars(args)
    json_str = json.dumps(args, indent=4)
    with open(path, 'w') as json_file:
        json_file.write(json_str)
        

def main():
    args = get_parser().parse_args()
    
    if not os.path.exists(args.fileModelSave):
        os.makedirs(args.fileModelSave)

    write_args_to_josn(args)

    model = init_model(args)
    print(model)

    tr_dataloader = init_dataloader(args, 'train')
    val_dataloader = init_dataloader(args, 'val')
    test_dataloader = init_dataloader(args, 'test')

    
    optim = init_optim(args, model)
    lr_scheduler = init_lr_scheduler(args, optim)
    results = train(args=args,
                    tr_dataloader=tr_dataloader,
                    val_dataloader=val_dataloader,
                    model=model,
                    optim=optim,
                    lr_scheduler=lr_scheduler)
    acc_best_state, f1_best_state, best_acc, train_loss, train_acc, val_loss, val_acc = results


    print('Testing with last model..')
    test(args=args,
         test_dataloader=test_dataloader,
         model=model)

    model.load_state_dict(acc_best_state)
    print('Testing with acc best model..')
    test(args=args,
         test_dataloader=test_dataloader,
         model=model)

    model.load_state_dict(f1_best_state)
    print('Testing with f1 best model..')
    test(args=args,
         test_dataloader=test_dataloader,
         model=model)

   
if __name__ == '__main__':
    main()
