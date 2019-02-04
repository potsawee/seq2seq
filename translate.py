import sys
import os
import numpy as np
import tensorflow as tf
import argparse
import pdb

from spellchecker import SpellChecker
from model import EncoderDecoder
from helper import load_vocab, read_config

'''
Translating source sentences to target sentences using a trained model
'''

def get_translate_arguments(parser):
    '''Arguments for translating'''

    parser.register("type", "bool", lambda v: v.lower() == "true")

    # file paths
    parser.add_argument('--load', type=str, required=True)  # path to load model
    parser.add_argument('--srcfile', type=str, required=True)
    parser.add_argument('--tgtfile', type=str, required=True)
    parser.add_argument('--spellcheck', type="bool", nargs="?", const=True, default=False)
    parser.add_argument('--model_number', type=int, default=None)

    return parser

def src_data(srcfile, src_word2id, max_sentence_length, spellcheck=False):
    src_sentences = []
    with open(srcfile, 'r') as file:
        for line in file:
            src_sentences.append(line.strip())

    if spellcheck:
        spell = SpellChecker()

    src_sent_ids = []
    for sentence in src_sentences:
        ids = []
        for word in sentence.split():
            if word in src_word2id:
                ids.append(src_word2id[word])
            else:
                if spellcheck:
                    x = spell.correction(word)
                    if x in src_word2id:
                        print("Spellcheck: {} => {}".format(word, x))
                        ids.append(src_word2id[x])
                    else:
                        ids.append(src_word2id['<unk>'])
                else:
                    ids.append(src_word2id['<unk>'])
        src_sent_ids.append(ids)

    # check if each sentence is too long
    for i in range(len(src_sent_ids)):
        if len(src_sent_ids[i]) > max_sentence_length:
            src_sent_ids[i] = src_sent_ids[i][:max_sentence_length]

    src_sent_len = [len(sent) for sent in src_sent_ids]
    src_sent_ids = [ids + [src_word2id['</s>']]*(max_sentence_length-len(ids)) for ids in src_sent_ids]

    return src_sent_ids, src_sent_len

def translate(config):
    if 'X_SGE_CUDA_DEVICE' in os.environ:
        print('running on the stack...')
        cuda_device = os.environ['X_SGE_CUDA_DEVICE']
        print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

    else: # development only e.g. air202
        print('running locally...')
        os.environ['CUDA_VISIBLE_DEVICES'] = '' # choose the device (GPU) here

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
    model_number = config['model_number'] if config['model_number'] != None else config['num_epochs'] - 1
    full_save_path_to_model = save_path + '/model-' + str(model_number)

    with tf.Session(config=sess_config) as sess:
        # Restore variables from disk.
        saver.restore(sess, full_save_path_to_model)
        # print("Model restored")

        src_sent_ids, src_sent_len = src_data(config['srcfile'], src_word2id,
                                            config['max_sentence_length'], config['spellcheck'])

        num_sentences = len(src_sent_ids)
        # batch_size = config['batch_size'] # maybe too small (inefficient) - but should be not too large
        batch_size = 100 # this is okay - it requires much lower memory compared to training
        num_batches = int(num_sentences/batch_size) + 1

        tgt_lines = []
        print('num_batches =', num_batches)

        for i in range(num_batches):

            i_start = batch_size*i
            i_end = i_start+batch_size if i_start+batch_size <= num_sentences else num_sentences
            translate_dict = {model.src_word_ids: src_sent_ids[i_start:i_end],
                        model.src_sentence_lengths: src_sent_len[i_start:i_end],
                        model.dropout: 0.0}

            [translations] = sess.run([model.translations], feed_dict=translate_dict)

            for translation in translations:
                words = []
                for id in translation:
                    if id == params['eos_id']:
                        break
                    words.append(tgt_id2word[id])

                # print(' '.join(words))
                tgt_lines.append(' '.join(words))

            print('#')
            sys.stdout.flush()

        with open(config['tgtfile'], 'w') as file:
            for line in tgt_lines:
                file.write(line + '\n')
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
    config['tgtfile'] = args['tgtfile']
    config['model_number'] = args['model_number']
    config['spellcheck'] = args['spellcheck']

    translate(config=config)

if __name__ == '__main__':
    main()
