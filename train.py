import os
import sys
import collections
import numpy as np
import tensorflow as tf
import argparse
import pdb

from model import EncoderDecoder

'''
Training the Encoder Decoder model
'''

def add_arguments(parser):
    '''Build Argument Parser'''
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # file paths
    parser.add_argument('--train_src', type=str, required=True)
    parser.add_argument('--train_tgt', type=str, required=True)
    parser.add_argument('--vocab_src', type=str, required=True)
    parser.add_argument('--vocab_tgt', type=str, required=True)
    parser.add_argument('--save', type=str, required=True) # path to save model
    parser.add_argument('--load', type=str, default=None)  # path to load model

    # network architecture
    parser.add_argument('--embedding_size', type=int, default=200)
    parser.add_argument('--num_units', type=int, default=128)

    # hyperpaprameters
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=256)

    # training settings
    parser.add_argument('--num_epochs', type=int, default=20)

    # data
    parser.add_argument('--max_sentence_length', type=int, default=32)

    # other settings
    parser.add_argument("--use_gpu", type="bool", nargs="?", const=True, default=False)

    return parser


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

def construct_training_data_batches(config):
    # train_src = 'data/iwslt15/train.en'
    # train_tgt = 'data/iwslt15/train.en'
    # # train_src = 'data/iwslt15/mytrain3.en'
    # # train_tgt = 'data/iwslt15/mytrain3.vi'
    # vocab_src = 'data/iwslt15/vocab.en'
    # vocab_tgt = 'data/iwslt15/vocab.en'

    train_src = config['train_src']
    train_tgt = config['train_tgt']
    vocab_src = config['vocab_src']
    vocab_tgt = config['vocab_tgt']

    batch_size = config['batch_size']
    max_sentence_length = config['max_sentence_length']

    vocab_paths = {'vocab_src': vocab_src, 'vocab_tgt': vocab_tgt}
    data_paths = {'train_src': train_src, 'train_tgt': train_tgt}

    src_word2id, tgt_word2id = load_vocab(vocab_paths)
    train_src_sentences, train_tgt_sentences = load_data(data_paths)

    vocab_size = {'src': len(src_word2id), 'tgt': len(tgt_word2id)}

    train_src_word_ids = [] # num_sentences x max_sentence_length
    train_tgt_word_ids = [] # num_sentences x max_sentence_length
    train_src_sentence_lengths = []
    train_tgt_sentence_lengths = []

    # EOS id
    src_eos_id = src_word2id['</s>']
    tgt_eos_id = tgt_word2id['</s>']

    # Source sentence
    for sentence in train_src_sentences:
        words = sentence.split()
        if len(words) > max_sentence_length:
            continue
        ids = [src_eos_id] * max_sentence_length
        for i, word in enumerate(words):
            if word in src_word2id:
                ids[i] = src_word2id[word]
            else:
                ids[i] = src_word2id['<unk>']
        train_src_word_ids.append(ids)
        train_src_sentence_lengths.append(len(words)+1) # include one EOS

    # Target sentence
    for sentence in train_tgt_sentences:
        words = sentence.split()
        if len(words) > max_sentence_length:
            continue
        ids = [tgt_eos_id] * max_sentence_length
        for i, word in enumerate(words):
            if word in tgt_word2id:
                ids[i] = tgt_word2id[word]
            else:
                ids[i] = tgt_word2id['<unk>']
        train_tgt_word_ids.append(ids)
        train_tgt_sentence_lengths.append(len(words)+1) # include one EOS


    assert (len(train_src_word_ids) == len(train_tgt_word_ids)), "train_src_word_ids != train_src_word_ids"
    num_training_sentences = len(train_src_word_ids)
    print("num_training_sentences: ", num_training_sentences) # only those that are not too long

    batches = []

    for i in range(int(num_training_sentences/batch_size)):
        i_start = i * batch_size
        i_end = i_start + batch_size
        batch = {'src_word_ids': train_src_word_ids[i_start:i_end],
            'tgt_word_ids': train_tgt_word_ids[i_start:i_end],
            'src_sentence_lengths': train_src_sentence_lengths[i_start:i_end],
            'tgt_sentence_lengths': train_tgt_sentence_lengths[i_start:i_end]}

        batches.append(batch)

    return batches, vocab_size, src_word2id, tgt_word2id

def train(config):
    # --------- configurations --------- #
    batch_size = config['batch_size']
    save_path = config['save'] # path to store model
    saved_model = config['load'] # None or path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # ---------------------------------- #
    write_config(save_path+'/config.txt', config)

    batches, vocab_size, src_word2id, tgt_word2id = construct_training_data_batches(config)

    tgt_id2word = list(tgt_word2id.keys())

    params = {'vocab_src_size': vocab_size['src'],
            'vocab_tgt_size': vocab_size['tgt'],
            'go_id':  tgt_word2id['<go>'],
            'eos_id':  tgt_word2id['</s>']}

    model = EncoderDecoder(config, params)
    model.build_network()

    # save & restore model
    saver = tf.train.Saver(max_to_keep=2)

    if config['use_gpu']:
        if 'X_SGE_CUDA_DEVICE' in os.environ:
            print('running on the stack...')
            cuda_device = os.environ['X_SGE_CUDA_DEVICE']
            print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

        else: # development only e.g. air202
            print('running locally...')
            os.environ['CUDA_VISIBLE_DEVICES'] = '1' # choose the device (GPU) here

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True # Whether the GPU memory usage can grow dynamically.
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.95 # The fraction of GPU memory that the process can use.

    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        sess_config = tf.ConfigProto()


    # sess = tf.Session(config=sess_config) # sess.close() / sess.run()
    with tf.Session(config=sess_config) as sess:

        if saved_model == None:
            sess.run(tf.global_variables_initializer())
        else:
            new_saver = tf.train.import_meta_graph(saved_model + '.meta')
            new_saver.restore(sess, saved_model)
            print('loaded model...', saved_model)


        # tf_variables = tf.trainable_variables()
        # for i in range(len(tf_variables)):
        #     print(tf_variables[i])

        num_epochs = config['num_epochs']
        for epoch in range(num_epochs):
            print("num_batches = ", len(batches))
            for i, batch in enumerate(batches):

                feed_dict = { model.src_word_ids: batch['src_word_ids'],
                            model.tgt_word_ids: batch['tgt_word_ids'],
                            model.src_sentence_lengths: batch['src_sentence_lengths'],
                            model.tgt_sentence_lengths: batch['tgt_sentence_lengths']}

                train_loss, _ = sess.run([model.train_loss, model.train_op],
                                    feed_dict=feed_dict)

                if i % 10 == 0:
                    infer_dict = { model.src_word_ids: batch['src_word_ids'],
                                model.tgt_word_ids: batch['tgt_word_ids'],
                                model.src_sentence_lengths: batch['src_sentence_lengths'],
                                model.tgt_sentence_lengths: batch['tgt_sentence_lengths']}

                    [my_translations, infer_loss] = sess.run([model.translations, model.infer_loss],
                                                        feed_dict=infer_dict)

                    print("batch: {} --- train_loss: {:.5f} | inf_loss: {:.5f}".format(i, train_loss, infer_loss))
                    sys.stdout.flush()

                if i % 50 == 0:

                    my_sentences = ['They wrote almost a thousand pages on the topic . </s>',
                                    'And it takes weeks to perform our integrations . </s>',
                                    'It was terribly dangerous . </s>',
                                    'This is a fourth alternative that you are soon going to have . </s>']
                    my_sent_ids = []

                    for my_sentence in my_sentences:
                        ids = []
                        for word in my_sentence.split():
                            if word in src_word2id:
                                ids.append(src_word2id[word])
                            else:
                                ids.append(src_word2id['<unk>'])
                        my_sent_ids.append(ids)

                    my_sent_len = [len(my_sent) for my_sent in my_sent_ids]
                    my_sent_ids = [ids + [src_word2id['</s>']]*(config['max_sentence_length']-len(ids)) for ids in my_sent_ids]


                    infer_dict = {model.src_word_ids: my_sent_ids,
                                model.src_sentence_lengths: my_sent_len}

                    [my_translations] = sess.run([model.translations], feed_dict=infer_dict)
                    # pdb.set_trace()
                    for my_sent in my_translations:
                        my_words = [tgt_id2word[id] for id in my_sent]
                        print(' '.join(my_words))


            print("################## EPOCH {} done ##################".format(epoch))
            saver.save(sess, save_path + '/model', global_step=epoch)

def write_config(path, config):
    with open(path, 'w') as file:
        for x in config:
            file.write('{}={}\n'.format(x, config[x]))
    print('write config done')


def main():
    # get configurations from the terminal
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = vars(parser.parse_args())

    train(config=args)

if __name__ == '__main__':
    main()
