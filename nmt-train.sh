#!/bin/bash
# souce = load the conda environment that includes
# python 3.6 bin path
# TensorFlow (GPU) 1.5.0
source /home/miproj/urop.2018/pm574/anaconda3/bin/activate tf_gpu

# python 3.6 bin path
# TensorFlow (CPU) 1.10.0
# source /home/miproj/urop.2018/pm574/anaconda3/bin/activate tf_cpu

python /home/alta/BLTSpeaking/ged-pm574/local/seq2seq/train.py \
    --train_src local/seq2seq/data/iwslt15/train.en \
    --train_tgt local/seq2seq/data/iwslt15/train.en \
    --vocab_src local/seq2seq/data/iwslt15/vocab.en \
    --vocab_tgt local/seq2seq/data/iwslt15/vocab.en \
    --save lib/models/tmp0 \
    --embedding_size 200 \
    --num_units 128 \
    --learning_rate 0.01 \
    --batch_size 256 \
    --num_epochs 1 \
    --max_sentence_length 32 \
    --use_gpu True