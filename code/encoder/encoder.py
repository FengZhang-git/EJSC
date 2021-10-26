

import math
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F


from transformers import BertTokenizer, BertConfig, BertModel, AdamW, AutoTokenizer, AutoModel


def parse_args():
    parser = argparse.ArgumentParser()
   
    parser.add_argument("--fileVocab", type=str)
    parser.add_argument("--fileModelConfig", type=str)
    parser.add_argument("--fileModel", type=str)
    parser.add_argument("--numDevice", type=int)
    parser.add_argument("--numFreeze", type=int)
    parser.add_argument("--numClass", type=int)


    args = parser.parse_args()
    return vars(args) 






class LSTMEncoder(nn.Module):
    

    def __init__(self, input_dim, hidden_dim, dropout_rate, extra_dim=None):
        

        super(LSTMEncoder, self).__init__()

        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate
        self.__extra_dim = extra_dim

      
        # Make sure the input dimension of iterative LSTM.
        
        if self.__extra_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim
        else:
            lstm_input_dim = self.__input_dim

        # Network parameter definition.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1
        )
       

    def forward(self, encoded_hiddens, seq_lens, forced_input=None, extra_input=None):
        

        # Concatenate information tensor if possible.
        if extra_input is not None:
            input_tensor = torch.cat([encoded_hiddens, extra_input], dim=2)
        else:
            input_tensor = encoded_hiddens

        dropout_input = self.__dropout_layer(input_tensor)
        lstm_out, _ = self.__lstm_layer(dropout_input)

        return lstm_out




class Extractor(nn.Module):
    

    def __init__(self, embedding_dim, dropout_rate, output_dim):
        super(Extractor, self).__init__()

      
        self.__embedding_dim = embedding_dim
        self.__dropout_rate = dropout_rate
        self.__output_dim = output_dim



        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__relu = nn.ReLU(inplace=True)
           

    def forward(self, embedded_text, embedding, seq_lens):
        

       
        dropout_text = self.__dropout_layer(embedded_text)
        embedding = embedding.mean(1)

        num_instance, max_token_len, dim = dropout_text.shape
        num_classes, dim1 = embedding.shape
        if dim != dim1:
            raise ValueError("Dim of class embedding is different form dim of query's")
        H = dropout_text
        E = embedding
        R_o = torch.matmul(H, (E.T).expand(num_instance, dim, num_classes))
        R_o = F.softmax(R_o, dim=2)
        H_o = torch.matmul(R_o, E.expand(num_instance, num_classes, dim))
        outputs = self.__relu(H_o)

        return outputs





class Encoder(nn.Module):

    def __init__(self, dropout_rate):
        

        super(Encoder, self).__init__()

        self.__dropout_rate = dropout_rate
        

        # Network parameter definition.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        
       

    def forward(self, encoded_hiddens, seq_lens, forced_input=None, extra_input=None):
        

        # Concatenate information tensor if possible.
        if extra_input is not None:
            input_tensor = torch.cat([encoded_hiddens, extra_input], dim=2)
        else:
            input_tensor = encoded_hiddens

        dropout_input = self.__dropout_layer(input_tensor)
    
        return dropout_input


class BertEncoder(nn.Module):

    def __init__(self, args):
        super(BertEncoder, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(args.fileVocab, do_basic_tokenize=False)
        config = BertConfig.from_json_file(args.fileModelConfig)   
        self.bert = BertModel.from_pretrained(args.fileModel,config=config)
        self.device = torch.device('cuda', args.numDevice)
        torch.cuda.set_device(self.device)

        if args.numFreeze > 0:
            self.freeze_layers(args.numFreeze)

        self.bi_lstm = LSTMEncoder(
            768, #word_embedding_dim,
            1536, #encoder_hidden_dim,
            0.1 #dropout_rate
        )

        self.bert.cuda()
        self.bi_lstm.cuda()


    def freeze_layers(self, numFreeze):
        unfreeze_layers = []
        for i in range(numFreeze, 12):
            unfreeze_layers.append("layer."+str(i))

        for name ,param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
    
    def forward(self, text):

        tokenizer = self.tokenizer(
        text,
        padding = True,
        truncation = True,
        max_length = 100,
        return_tensors='pt'  # 返回的类型为pytorch tensor
        )
        seq_lens = []
        


        input_ids = tokenizer['input_ids'].to(self.device)
        token_type_ids = tokenizer['token_type_ids'].to(self.device)
        attention_mask = tokenizer['attention_mask'].to(self.device)

        

        outputs = self.bert(
              input_ids,
              attention_mask=attention_mask,
              token_type_ids=token_type_ids
              )
        token_word_tensor, cls_embedding = outputs
        bilstm_hiddens = self.bi_lstm(token_word_tensor, seq_lens)
       

        return bilstm_hiddens, seq_lens
      
       

class ModelManager(nn.Module):

    def __init__(self, args):
        super(ModelManager, self).__init__()

        self.__encoder = BertEncoder(args)
        self.__slot__extractor = Extractor(
            embedding_dim = 1536*2,
            dropout_rate = args.dropout_rate,
            output_dim = 768
        )
        self.__intent_extractor = Extractor(
            embedding_dim = 1536*2,
            dropout_rate = args.dropout_rate,
            output_dim = 768
        )
        self.__slot_encoder = Encoder(
            dropout_rate = args.dropout_rate
            )
        self.__intent_encoder = Encoder(
            dropout_rate = args.dropout_rate
            )
        

    def forward(self, text, intent_description, slots_description, seq_lens=None, n_predicts=None, forced_slot=None, forced_intent=None):
        word_tensor, seq_lens = self.__encoder(text)
        

        intent_embedding, x = self.__encoder(intent_description)
        slots_embedding, y = self.__encoder(slots_description)

        

        slot_features = self.__slot__extractor(word_tensor, slots_embedding, seq_lens)
        intent_features = self.__intent_extractor(word_tensor, intent_embedding, seq_lens)
      
        pred_intent = self.__intent_encoder(
            word_tensor, seq_lens, extra_input=slot_features
        )
        
        pred_slot = self.__slot_encoder(
            word_tensor, seq_lens, extra_input=intent_features
        )


        return word_tensor, pred_intent, pred_slot


   # visiualization
    def get_word_embeddings(self, text):
         word_tensor = self.__encoder(text)

         return word_tensor
