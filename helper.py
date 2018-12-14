import collections

<<<<<<< HEAD
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import dtypes
from tensorflow.contrib.seq2seq import SampleEmbeddingHelper

=======
>>>>>>> 35bd4f2c4754a1bfd90631f93dbc22ab0498bf7f
'''
Functions for train & translate
'''

def load_vocab(paths):
    with open(paths['vocab_src']) as file:
        vocab_src = file.readlines()
    with open(paths['vocab_tgt']) as file:
        vocab_tgt = file.readlines()

    src_word2id = collections.OrderedDict()
    tgt_word2id = collections.OrderedDict()

    for i, word in enumerate(vocab_src):
        word = word.strip() # remove \n
        src_word2id[word] = i

    for i, word in enumerate(vocab_tgt):
        word = word.strip() # remove \n
        tgt_word2id[word] = i

    # -------------- Special Tokens -------------- #
    # <go> <unk> </s> are defined in the vocab list
    # -------------------------------------------- #

    return src_word2id, tgt_word2id

def load_data(paths):
    with open(paths['train_src']) as file:
        train_src_sentences = file.readlines()
    with open(paths['train_tgt']) as file:
        train_tgt_sentences = file.readlines()

    assert (len(train_src_sentences) == len(train_tgt_sentences)), "train_source != train_target"
    # print("num_training_sentences: ", len(train_src_sentences))

    return train_src_sentences, train_tgt_sentences

def write_config(path, config):
    with open(path, 'w') as file:
        for x in config:
            file.write('{}={}\n'.format(x, config[x]))
    # print('write config done')

def read_config(path):
    config = {}
    with open(path, 'r') as file:
        for line in file:
            x = line.strip().split('=')
            key = x[0]
            if x[1].isdigit():
                val = int(x[1])
            elif isfloat(x[1]):
                val = float(x[1])
            else: # string
                val = x[1]

            config[key] = val

    # print('read config done')
    return config

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False
<<<<<<< HEAD

'''
Helper classes
'''

class ModifiedSampleEmbeddingHelper(SampleEmbeddingHelper):
    """
    sample: same as SampleEmbeddingHelper
    next_inputs: get from argmax (i.e. GreedyEmbeddingHelper)
    """

    def __init__(self, embedding, start_tokens, end_token, softmax_temperature=None, seed=None):
        super(ModifiedSampleEmbeddingHelper, self).__init__(
            embedding, start_tokens, end_token,
            softmax_temperature, seed)

    # modified!
    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for ModifiedSampleEmbeddingHelper."""
        del time, sample_ids  # unused by next_inputs_fn

        argmax_ids = math_ops.argmax(outputs, axis=-1, output_type=dtypes.int32)

        finished = math_ops.equal(argmax_ids, self._end_token)
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self._start_inputs,
            lambda: self._embedding_fn(argmax_ids))
        return (finished, next_inputs, state)
=======
>>>>>>> 35bd4f2c4754a1bfd90631f93dbc22ab0498bf7f
