
seq2seq model
=====================================================
Requirements
--------------------------------------
- python 3.6
- TensorFlow 1.5.0

example:

    source /home/miproj/urop.2018/pm574/anaconda3/bin/activate tf_gpu

Encoder&Decoder for GED/GEC experiments
--------------------------------------
Training:

    python train.py \
        --train_src data/iwslt15/train.en \
        --train_tgt data/iwslt15/train.en \
        --vocab_src data/iwslt15/vocab.en \
        --vocab_tgt data/iwslt15/vocab.en \
        --save lib/models/tmp0 \
        --embedding_size 200 \
        --num_units 128 \
        --learning_rate 0.01 \
        --batch_size 256 \
        --num_epochs 1 \
        --max_sentence_length 32 \
        --use_gpu True

Translating:
