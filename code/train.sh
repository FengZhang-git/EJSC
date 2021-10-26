#!/bin/bash


python train.py \
            --dataFile "/data/atis/data" \
            --vocabFile "/data/atis/vocab" \
            --fileVocab "/code/pre-trained/bert_base_uncased/bert-base-uncased-vocab.txt" \
            --fileModelConfig "/code/pre-trained/bert_base_uncased/bert-base-uncased-config.json" \
            --fileModel "/code/pre-trained/bert_base_uncased/bert-base-uncased-pytorch_model.bin" \
            --fileModelSave "/model/atis" \
            --numDevice 1 \
            --learning_rate 1e-4 \
            --epochs 20 \
            --num_episode_per_epoch 100 \
            --num_episode_val 100 \
            --num_episode_test 100 \
            --numFreeze 6 \
            --warmup_steps 100 \
            --dropout_rate 0.1 \
            --weight_decay 0.2 \
            --temperature 0 \
            --lamda1 0.7 \
            --lamda2 0 \
            --lamda3 0

