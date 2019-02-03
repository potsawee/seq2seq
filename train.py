import os
import sys
import random
# import numpy as np
import tensorflow as tf
import argparse
import math
import pdb

from model import EncoderDecoder
from helper import load_vocab, load_data, write_config, load_pretrained_embedding

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
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_units', type=int, default=128)

    # hyperpaprameters
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--decay_rate', type=float, default=0.999)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--beam_width', type=int, default=10) # only use if decoding_method == beamsearch

    # training settings
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--random_seed', type=int, default=25)
    parser.add_argument('--decoding_method', type=str, default='greedy')
    parser.add_argument('--scheduled_sampling', type="bool", nargs="?", const=True, default=False)
    parser.add_argument('--residual', type="bool", nargs="?", const=True, default=False)
    parser.add_argument('--load_embedding_src', type=str, default=None)
    parser.add_argument('--load_embedding_tgt', type=str, default=None)


    # data
    parser.add_argument('--max_sentence_length', type=int, default=32)

    # other settings
    parser.add_argument("--use_gpu", type="bool", nargs="?", const=True, default=False)

    return parser

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
    print("num_vocab_src: ", vocab_size['src'])
    print("num_vocab_tgt: ", vocab_size['tgt'])

    train_src_word_ids = [] # num_sentences x max_sentence_length
    train_tgt_word_ids = [] # num_sentences x max_sentence_length
    train_src_sentence_lengths = []
    train_tgt_sentence_lengths = []

    # EOS id
    src_eos_id = src_word2id['</s>']
    tgt_eos_id = tgt_word2id['</s>']

    # Source and Target sentences
    for src_sentence, tgt_sentence in zip(train_src_sentences, train_tgt_sentences):
        src_words = src_sentence.split()
        tgt_words = tgt_sentence.split()

        if len(src_words) > max_sentence_length or len(tgt_words) > max_sentence_length:
            continue

        # source
        src_ids = [src_eos_id] * max_sentence_length
        for i, word in enumerate(src_words):
            if word in src_word2id:
                src_ids[i] = src_word2id[word]
            else:
                src_ids[i] = src_word2id['<unk>']
        train_src_word_ids.append(src_ids)
        train_src_sentence_lengths.append(len(src_words)+1) # include one EOS

        # target
        tgt_ids = [tgt_eos_id] * max_sentence_length
        for i, word in enumerate(tgt_words):
            if word in tgt_word2id:
                tgt_ids[i] = tgt_word2id[word]
            else:
                tgt_ids[i] = tgt_word2id['<unk>']
        train_tgt_word_ids.append(tgt_ids)
        train_tgt_sentence_lengths.append(len(tgt_words)+1) # include one EOS

    assert (len(train_src_word_ids) == len(train_tgt_word_ids)), "train_src_word_ids != train_src_word_ids"
    num_training_sentences = len(train_src_word_ids)
    print("num_training_sentences: ", num_training_sentences) # only those that are not too long

    # shuffle
    _x = list(zip(train_src_word_ids, train_tgt_word_ids, train_src_sentence_lengths, train_tgt_sentence_lengths))
    random.shuffle(_x)
    train_src_word_ids, train_tgt_word_ids, train_src_sentence_lengths, train_tgt_sentence_lengths = zip(*_x)

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

    # random seed
    random.seed(config['random_seed'])
    # np.random.seed(config['random_seed'])

    batches, vocab_size, src_word2id, tgt_word2id = construct_training_data_batches(config)

    tgt_id2word = list(tgt_word2id.keys())

    params = {'vocab_src_size': vocab_size['src'],
            'vocab_tgt_size': vocab_size['tgt'],
            'go_id':  tgt_word2id['<go>'],
            'eos_id':  tgt_word2id['</s>']}

    model = EncoderDecoder(config, params)
    model.build_network()

    learning_rate = config['learning_rate']
    decay_rate = config['decay_rate']

    tf_variables = tf.trainable_variables()
    for i in range(len(tf_variables)):
        print(tf_variables[i])

    # save & restore model
    saver = tf.train.Saver(max_to_keep=1)

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


    with tf.Session(config=sess_config) as sess:

        if saved_model == None:
            sess.run(tf.global_variables_initializer())
            # ------------ load pre-trained embeddings ------------ #
            if config['load_embedding_src'] != None:
                src_embedding = sess.run(model.src_word_embeddings)
                src_embedding_matrix = load_pretrained_embedding(src_word2id, src_embedding, config['load_embedding_src'])
                sess.run(model.src_word_embeddings.assign(src_embedding_matrix))
            if config['load_embedding_tgt'] != None:
                if config['load_embedding_tgt'] == config['load_embedding_src']:
                    sess.run(model.tgt_word_embeddings.assign(src_embedding_matrix))
                else:
                    tgt_embedding = sess.run(model.tgt_word_embeddings)
                    tgt_embedding_matrix = load_pretrained_embedding(tgt_word2id, tgt_embedding, config['load_embedding_tgt'])
                    sess.run(model.tgt_word_embeddings.assign(tgt_embedding_matrix))
            # ----------------------------------------------------- #
        else:
            new_saver = tf.train.import_meta_graph(saved_model + '.meta')
            new_saver.restore(sess, saved_model)
            print('loaded model...', saved_model)

        # ------------ TensorBoard ------------ #
        # summary_writer = tf.summary.FileWriter(save_path + '/tfboard/', graph_def=sess.graph_def)
        # ------------------------------------- #

        # ------------ To print out some output -------------------- #
        my_sentences = ['this is test . </s>',
                        'this is confirm my reservation at hotel . </s>',
                        'playing tennis good for you . </s>',
                        'when talking about successful longterm business relationships customer services are important element </s>']
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
                    model.src_sentence_lengths: my_sent_len,
                    model.dropout: 0.0,
                    model.learning_rate: learning_rate}
        # ---------------------------------------------------------- #

        num_epochs = config['num_epochs']
        for epoch in range(num_epochs):
            print("num_batches = ", len(batches))

            random.shuffle(batches)

            epoch_loss = 0

            for i, batch in enumerate(batches):

                feed_dict = { model.src_word_ids: batch['src_word_ids'],
                            model.tgt_word_ids: batch['tgt_word_ids'],
                            model.src_sentence_lengths: batch['src_sentence_lengths'],
                            model.tgt_sentence_lengths: batch['tgt_sentence_lengths'],
                            model.dropout: config['dropout'],
                            model.learning_rate: learning_rate}

                [_, loss] = sess.run([model.train_op, model.train_loss], feed_dict=feed_dict)
                epoch_loss += loss

                if i % 100 == 0:
                    # to print out training status

                    # if config['decoding_method'] != 'beamsearch':
                        # [train_loss, infer_loss] = sess.run([model.train_loss, model.infer_loss], feed_dict=feed_dict)
                        # print("batch: {} --- train_loss: {:.5f} | inf_loss: {:.5f}".format(i, train_loss, infer_loss))

                    # else:
                        # --- beam search --- #
                        # [train_loss] = sess.run([model.train_loss], feed_dict=feed_dict)
                        # print("BEAMSEARCH - batch: {} --- train_loss: {:.5f}".format(i, train_loss))

                    print("batch: {} --- avg train loss: {:.5f}".format(i, epoch_loss/(i+1)))

                    sys.stdout.flush()

                if i % 500 == 0:

                    [my_translations] = sess.run([model.translations], feed_dict=infer_dict)
                    # pdb.set_trace()
                    for my_sent in my_translations:
                        my_words = [tgt_id2word[id] for id in my_sent]
                        print(' '.join(my_words))

            model.increment_counter()
            learning_rate *= decay_rate

            print("---------------------------------------------------")
            print("epoch {} done".format(epoch+1))
            print("total training loss = {}".format(epoch_loss))
            print("---------------------------------------------------")

            if math.isnan(epoch_loss):
                print("stop training - loss/gradient exploded")
                break

            saver.save(sess, save_path + '/model', global_step=epoch)

def main():
    # get configurations from the terminal
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = vars(parser.parse_args())

    train(config=args)

if __name__ == '__main__':
    main()
