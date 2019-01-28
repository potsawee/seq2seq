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
    --train_src legacy/nmt-google/lib/data_ged/CLC.tgt \
    --train_tgt legacy/nmt-google/lib/data_ged/CLC.src \
    --vocab_src lib/wlists/vocab.clc.min-count2.en \
    --vocab_tgt lib/wlists/vocab.clc.min-count2.en \
    --embedding_size 200 \
    --num_layers 8 \
    --dropout 0.2 \
    --num_units 200 \
    --learning_rate 0.001 \
    --batch_size 256 \
    --num_epochs 10 \
    --random_seed 25 \
    --decoding_method beamsearch \
    --beam_width 10 \
    --max_sentence_length 50 \
    --scheduled_sampling True \
    --residual True \
    --use_gpu True \
    --save lib/models/residual/deep8 \
    --load_embedding_src lib/embeddings/glove.6B.200d.txt \
    --load_embedding_tgt lib/embeddings/glove.6B.200d.txt \
