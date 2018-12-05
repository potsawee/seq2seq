#!/bin/bash

qsub -cwd -j yes -o LOGs/train-en-v1.log \
    -P esol \
    -l qp=cuda_low \
    -l osrel='*' \
    -l mem_grab=60G -l mem_free=60G \
    /home/alta/BLTSpeaking/ged-pm574/local/seq2seq/nmt-exp.sh
