
seq2seq model
=====================================================
Requirements
--------------------------------------
- python 3.6
- TensorFlow 1.5.0

Encoder&Decoder for GED/GEC experiments
--------------------------------------
**Training**:

    python train.py \
        --train_src lib/data/source.txt \
        --train_tgt lib/data/target.txt \
        --vocab_src lib/wlists/source.txt \
        --vocab_tgt lib/wlists/target.txt \
        --embedding_size 200 \
        --load_embedding_src lib/embeddings/glove.6B.200d.txt \
        --load_embedding_tgt lib/embeddings/glove.6B.200d.txt \
        --num_layers 2 \
        --dropout 0.2 \
        --num_units 128 \
        --learning_rate 0.001 \
        --batch_size 256 \
        --num_epochs 100 \
        --random_seed 25 \
        --decoding_method beamsearch \
        --beam_width 10 \
        --max_sentence_length 32 \
        --scheduled_sampling True \
        --use_gpu True \
        --save lib/models/tmp0

**Translating**:

    python translate.py \
        --load lib/models/tmp0 \
        --srcfile lib/srcfile.txt \ 
        --tgtfile lib/tgtfile.txt \ 
        --model_number 19
        
Configurations
--------------------------------------
Change these configurations in the run/nmt-train.sh before training, and it will be saved at save_path/config.txt
- **train_src** - path to source data (one line per sentence)
- **train_tgt** - path to target data (one line per sentence)
- **vocab_src** - path to source vocabulary (one line per word including <go>, <unk>, and </s>)
- **vocab_src** - path to target vocabulary (one line per word including <go>, <unk>, and </s>)
- **embedding_size** - embedding layer size (default 200)
- **load_embedding_src** - path to pre-trained embedding for the encoder (default None)
- **load_embedding_tgt** - path to pre-trained embedding for the decoder (default None)
- **num_layers** - the number of LSTM layers (must be even as the encoder is bi-directional, default 2)
- **dropout** - the dropout probability of the LSTM layers (default 0.0)
- **num_units** - the number of the hidden units (default 128)
- **learning_rate** - learning rate (default 0.01)
- **batch_size** - batch size (default 256)
- **random_seed** - random seed (default 25)
- **decoding_method** - greedy/sample1/sample2/beamsearch
- **beam_width** - beam width (default 10) only used if decoding_method == beamsearch
- **max_sentence_length** - sentences longer than this will be neglected (default 32)
- **scheduled_sampling** - True/False whether to enable Scheduled Sampling for training
- **use_gpu** - enable GPU (default True)
- **save** - path to the location to store the trained model & the configurations
        
Example (running on a CUED machine)
--------------------------------------
1. Installing & Setting up the environment using conda 

        conda create --name tf_gpu tensorflow-gpu python=3.6
        source activate tf_gpu
        conda install -c conda-forge tensorflow=1.5
        conda install -c conda-forge cudatoolkit=8.0
        
    note that you may need the full path to the conda binary, and the full path the activate script under anaconda3/bin/
    
2. Training an NMT system

    2.1) Change in run/nmt-train.sh
    
        export PATH=<your-anaconda-bin-path>:$PATH
        export PYTHONBIN=<your-python-bin-path>
    
    2.2) Set the configurations in the **run/nmt-train.sh** script
    
    2.3) Run
    
        locally:  ./run/nmt-train.sh
        on stack: ./run/run-nmt-train.sh log
    
3. Translating using a trained system

    3.1) Change in run/nmt-translate.sh
    
        export PATH=<your-anaconda-bin-path>:$PATH
        export PYTHONBIN=<your-python-bin-path>
            
    3.2) Set the configurations in the **run/nmt-translate.sh** script
    
    - load - path to trained model
    - srcfile - source file to be translated (one sentence per line)
    - tgtfile - target file (output)

    3.3) Run
    
        locally:  ./run/nmt-translate.sh
        on stack: ./run/run-nmt-translate.sh log        
  

