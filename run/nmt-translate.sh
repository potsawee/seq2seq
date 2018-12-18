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

$PYTHONBIN /home/alta/BLTSpeaking/ged-pm574/local/seq2seq/translate.py \
    --load lib/models/clc-arf-sample1 \
    --srcfile /home/alta/BLTSpeaking/ged-pm574/artificial-error/lib/nmt-corrupt/cts8.gec.orig.dot \
    --tgtfile /home/alta/BLTSpeaking/ged-pm574/artificial-error/lib/nmt-corrupt/clc-arf-sample1/cts8.gec.nmt.dot
    # --model_number
