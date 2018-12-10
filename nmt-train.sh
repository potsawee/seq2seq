#!/bin/bash
#$ -S /bin/bash

unset LD_PRELOAD # when run on the stack it has /usr/local/grid/agile/admin/cudadevice.so which will give core dumped
export PATH=/home/miproj/urop.2018/pm574/anaconda3/bin/:$PATH # to use activate

# souce = load the conda environment that includes
# python 3.6 bin path
# TensorFlow (GPU) 1.5.0
source activate tf_gpu

# python 3.6 bin path
# TensorFlow (CPU) 1.10.0
# source activate tf_cpu

export PYTHONBIN=/home/miproj/urop.2018/pm574/anaconda3/envs/tf_gpu/bin/python

$PYTHONBIN /home/alta/BLTSpeaking/ged-pm574/local/seq2seq/train.py \
    --train_src local/seq2seq/data/iwslt15/train.en \
    --train_tgt local/seq2seq/data/iwslt15/train.en \
    --vocab_src local/seq2seq/data/iwslt15/vocab.en \
    --vocab_tgt local/seq2seq/data/iwslt15/vocab.en \
    --embedding_size 200 \
    --num_layers 2 \
    --dropout 0.2 \
    --num_units 128 \
    --learning_rate 0.001 \
    --batch_size 256 \
    --num_epochs 100 \
    --random_seed 25 \
    --decoding_method greedy \
    --max_sentence_length 32 \
    --use_gpu True \
    --save lib/models/tmp0
