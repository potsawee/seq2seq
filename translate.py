import os
import tensorflow as tf
import argparse
import pdb

from model import EncoderDecoder
from helper import load_vocab, read_config

'''
Translating source sentences to target sentences using a trained model
'''

def get_translate_arguments(parser):
    '''Arguments for translating'''
    # file paths
    parser.add_argument('--srcfile', type=str, required=True)
    parser.add_argument('--load', type=str, required=True)  # path to load model

    return parser

def src_data(srcfile, src_word2id, max_sentence_length):
    src_sentences = []
    with open(srcfile, 'r') as file:
        for line in file:
            src_sentences.append(line.strip())

    src_sent_ids = []
    for sentence in src_sentences:
        ids = []
        for word in sentence.split():
            if word in src_word2id:
                ids.append(src_word2id[word])
            else:
                ids.append(src_word2id['<unk>'])
        src_sent_ids.append(ids)

    src_sent_len = [len(sent) for sent in src_sent_ids]
    src_sent_ids = [ids + [src_word2id['</s>']]*(max_sentence_length-len(ids)) for ids in src_sent_ids]

    return src_sent_ids, src_sent_len

def translate(config):
    # Running on a CPU should be fine
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sess_config = tf.ConfigProto()

    vocab_paths = {'vocab_src': config['vocab_src'], 'vocab_tgt': config['vocab_tgt']}
    src_word2id, tgt_word2id = load_vocab(vocab_paths)

    tgt_id2word = list(tgt_word2id.keys())

    params = {'vocab_src_size': len(src_word2id),
            'vocab_tgt_size': len(tgt_word2id),
            'go_id':  tgt_word2id['<go>'],
            'eos_id':  tgt_word2id['</s>']}

    # build the model
    model = EncoderDecoder(config, params)
    model.build_network()

    # save & restore model
    saver = tf.train.Saver()
    save_path = config['load']
    model_number = config['num_epochs'] - 1
    full_save_path_to_model = save_path + '/model-' + str(model_number)

    with tf.Session(config=sess_config) as sess:
        # Restore variables from disk.
        saver.restore(sess, full_save_path_to_model)
        print("Model restored")

        src_sent_ids, src_sent_len = src_data(config['srcfile'], src_word2id, config['max_sentence_length'])

        translate_dict = {model.src_word_ids: src_sent_ids,
                    model.src_sentence_lengths: src_sent_len,
                    model.dropout: 0.0}

        [translations] = sess.run([model.translations], feed_dict=translate_dict)

        for translation in translations:
            words = [tgt_id2word[id] for id in translation]
            print(' '.join(words))

        print('translation done!')

def main():
    # get configurations from the terminal
    parser = argparse.ArgumentParser()
    parser = get_translate_arguments(parser)
    args = vars(parser.parse_args())

    config_path = args['load'] + '/config.txt'
    config = read_config(config_path)
    config['load'] = args['load']
    config['srcfile'] = args['srcfile']

    translate(config=config)

if __name__ == '__main__':
    main()
