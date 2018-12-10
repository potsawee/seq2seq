import collections

'''
Functions for train & translate
'''

def load_vocab(paths):
    with open(paths['vocab_src']) as file:
        vocab_src = file.readlines()
    with open(paths['vocab_tgt']) as file:
        vocab_tgt = file.readlines()

    print("num_vocab_src: ", len(vocab_src))
    print("num_vocab_tgt: ", len(vocab_tgt))

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
    print('write config done')

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

    print('read config done')
    return config

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False
