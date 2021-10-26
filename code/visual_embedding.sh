
python visual_embedding.py \
            --dataFile "/data/top/data" \
            --vocabFile "/data/top/vocab" \
            --fileVocab "/pre-trained/bert_base_uncased/bert-base-uncased-vocab.txt" \
            --fileModelConfig "/pre-trained/bert_base_uncased/bert-base-uncased-config.json" \
            --fileModel "/pre-trained/bert_base_uncased/bert-base-uncased-pytorch_model.bin" \
            --fileModelSave "/model/top/model" \
            --numDevice 0 \
            --learning_rate 1e-4 \
            --numFreeze 6 \
            --warmup_steps 100 \
            --dropout_rate 0.1 \
            --weight_decay 0.2 \
            --temperature 0 \
            --lamda 0.7 \
            --gamma 0 \
            --delta 0 


