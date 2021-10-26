# README

Code for

Liu, H., Zhang, F., Zhang, X.T., Zhao, S.Y., & Zhang, X.C.(2021, November). An Explicit-Joint and Supervised-Contrastive Learning Framework for Few-Shot Intent Classification and Slot Filling. In Findings of the 2021 Conference on Empirical Methods in Natural Language Processing(Findings of EMNLP)

## Platform

Our experiments are conducted on a platform with Intel(R) Xeon(R) Gold 6248R CPU @3.00GHz and single GPU NVIDIA TITAN RTX 24GB.

## Environment

```
conda env create -f environment.yaml
```

## Running

```
cd code
bash train.sh
```

The detailed configurations can be found in the ```train.sh```. As the Bert model is too large, you can download the Bert model from [Hugging Face(```bert-base-uncased```)](https://huggingface.co/bert-base-uncased).

## Files Definition

- ```data``` : contains three public datasets: SNIPS, ATIS and TOP

- ```code``` : contains python files of our framework

    - ```data_process``` : used to sample each episode's data
    - ```encoder``` : model file
    - ```losses.py``` : contains loss function
    - ```parser_util.py``` : parse parameters
    - ```train.py``` : train the model
    - ```train.sh``` : parameters used to train models
    - ```visual_embedding.py``` : visualize word embeddings
    - ```visual_embedding.sh```: parameters used to visualize word embeddings  
 
