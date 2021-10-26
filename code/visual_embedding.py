import os
import pprint
import numpy as np
from numpy.lib.function_base import append
import json
from tqdm import tqdm
from encoder.encoder import ModelManager
import torch
from collections import defaultdict

from parser_util import get_parser
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def get_data(path, mode=None):
    result = []
    with open(path, 'r') as src:
        if mode == 'json':
            for line in tqdm(src):
                line = json.loads(line)
                result.append(line)
        else:
            for line in tqdm(src):
                line = line.split('\n')[0]
                result.append(line)
    return result

def get_base_model(args):
    model = ModelManager(args).to(device)
    return model
   
def get_trained_model(args, modelPath):
    model = ModelManager(args).to(device)
    model.load_state_dict(torch.load(modelPath))
    return model
   

if __name__ == '__main__':
    args = get_parser().parse_args()
    mat = []
    labels = []
    modelPath = "/model/top/best_model.pth"
    dataPath = "/top/data/train.json"
    savePath = '/visual/basefigure.jpg'
   
    datas = get_data(dataPath, mode='json')
    text = []
    for line in datas:
        text.append(line["text_u"])
        labels.append([line["text_u"], line["intent"]])

    device = torch.device('cuda', args.numDevice)
    torch.cuda.set_device(device)
    model = get_base_model(args)
    # model = get_trained_model(args, modelPath)

    for name, param in model.named_parameters():
        param.requires_grad = False
    
    
    bsz = 100
    batch = int(len(text)/bsz) + 1
    for i in range(batch):
       
        if (i+1)*bsz >= len(text):
            text_u = text[i*bsz:]
        else:
            text_u = text[i*bsz: (i+1)*bsz]

        pooled_output = model.get_word_embeddings(text_u)
        outputs = pooled_output.mean(dim=1)
        mat.extend(outputs)
    

    matrix_mat = torch.stack(mat).squeeze(1)
    print(matrix_mat.shape)
    X = matrix_mat.cpu().numpy()

    pca = PCA(n_components=2)
    pca.fit(X)
    X_new = pca.transform(X)
    mydict = defaultdict(list)
   
    for i, item in enumerate(X_new):
        item = list(item)
        mydict[labels[i][1]].append(item)
        
    
    plt.rcParams['savefig.dpi'] = 300 
    plt.rcParams['figure.dpi'] = 300 
    colors = ['blue', 'green', 'red', 'm', 'yellow', 'black', 'c']
    for i, key in enumerate(mydict.keys()):
        plt.scatter(np.array(mydict[key])[:, 0], np.array(mydict[key])[:, 1], s = 2, alpha = 0.4, marker='.', color=colors[i])
   
    plt.grid()
    plt.axis('off')
    plt.savefig(savePath)
    plt.show()
   
    