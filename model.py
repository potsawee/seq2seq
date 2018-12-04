import os
import collections
import numpy as np
import tensorflow as tf
import pdb

'''
Encoder-Decoder architecture
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
        # src_word2id[word] = i+1 # start from 1
        src_word2id[word] = i

    for i, word in enumerate(vocab_tgt):
        word = word.strip() # remove \n
        # tgt_word2id[word] = i+1 # start from 1
        tgt_word2id[word] = i


    # -------------- Special Tokens -------------- #
    # <unk>, <s>, </s> are defined in the vocab list
    # -------------------------------------------- #

    return src_word2id, tgt_word2id

def load_data(paths):
    with open(paths['train_src']) as file:
        train_src_sentences = file.readlines()
    with open(paths['train_tgt']) as file:
        train_tgt_sentences = file.readlines()

    assert (len(train_src_sentences) == len(train_tgt_sentences)), "train_source != train_target"
    print("num_training_sentences: ", len(train_src_sentences))

    return train_src_sentences, train_tgt_sentences


def construct_training_data_batches():
    train_src = 'data/iwslt15/train.en'
    train_tgt = 'data/iwslt15/train.en'
    # train_src = 'data/iwslt15/mytrain3.en'
    # train_tgt = 'data/iwslt15/mytrain3.vi'
    vocab_src = 'data/iwslt15/vocab.en'
    vocab_tgt = 'data/iwslt15/vocab.en'

    vocab_paths = {'vocab_src': vocab_src, 'vocab_tgt': vocab_tgt}
    data_paths = {'train_src': train_src, 'train_tgt': train_tgt}

    src_word2id, tgt_word2id = load_vocab(vocab_paths)
    train_src_sentences, train_tgt_sentences = load_data(data_paths)

    vocab_size = {'src': len(src_word2id), 'tgt': len(tgt_word2id)}
    num_training_sentences = len(train_src_sentences)

    max_sentence_length = 32

    train_src_word_ids = [] # num_sentences x max_sentence_length
    train_tgt_word_ids = [] # num_sentences x max_sentence_length
    train_src_sentence_lengths = []
    train_tgt_sentence_lengths = []

    # Source sentence
    for sentence in train_src_sentences:
        words = sentence.split()
        if len(words) > max_sentence_length:
            continue
        ids = [src_word2id['</s>']] * max_sentence_length
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
        ids = [src_word2id['</s>']] * max_sentence_length
        for i, word in enumerate(words):
            if word in tgt_word2id:
                ids[i] = tgt_word2id[word]
            else:
                ids[i] = tgt_word2id['<unk>']
        train_tgt_word_ids.append(ids)
        train_tgt_sentence_lengths.append(len(words)+1) # include one EOS


    batch_size = 256
    batches = []
    for i in range(int(num_training_sentences/batch_size)-1):
        batch = {'src_word_ids': train_src_word_ids[i:i+batch_size],
            'tgt_word_ids': train_tgt_word_ids[i:i+batch_size],
            'src_sentence_lengths': train_src_sentence_lengths[i:i+batch_size],
            'tgt_sentence_lengths': train_tgt_sentence_lengths[i:i+batch_size]}

        batches.append(batch)

    return batches, vocab_size, src_word2id, tgt_word2id



class EncoderDecoder(object):
    def __init__(self, params):
        # hyper-parameters / configurations
        self.learning_rate = 0.01
        self.batch_size = 256
        self.embedding_size = 200
        self.num_units = 128
        self.max_sentence_length = 32
        self.params = params


    def build_network(self):

    ################### placeholders
        self.src_word_ids = tf.placeholder(tf.int32, [None, None], name="src_word_ids")
        self.tgt_word_ids = tf.placeholder(tf.int32, [None, None], name="tgt_word_ids")
        self.src_sentence_lengths = tf.placeholder(tf.int32, [None], name="src_sentence_lengths")
        self.tgt_sentence_lengths = tf.placeholder(tf.int32, [None], name="tgt_sentence_lengths")

    ################### padding <go>
        go_id = self.params['go_id']
        self.tgt_word_ids_with_go = tf.concat( [tf.fill([self.batch_size, 1], go_id), self.tgt_word_ids], 1)
    ################### embeddings

        self.src_word_embeddings = tf.get_variable("src_word_embeddings",
                                shape=[self.params['vocab_src_size'], self.embedding_size],
                                initializer=tf.glorot_normal_initializer())
        self.tgt_word_embeddings = tf.get_variable("tgt_word_embeddings",
                                shape=[self.params['vocab_tgt_size'], self.embedding_size],
                                initializer=tf.glorot_normal_initializer())


        self.src_embedded = tf.nn.embedding_lookup(self.src_word_embeddings, self.src_word_ids)
        self.tgt_embedded = tf.nn.embedding_lookup(self.tgt_word_embeddings, self.tgt_word_ids_with_go)
        # Look up embedding:
        #   encoder_inputs:  [batch_size, max_time]
        #   encoder_emb_inp: [batch_size, max_time, embedding_size]

        s = tf.shape(self.src_word_ids) # s[0] = batch_size , s[1] = max_sentecce_length

    ################### def build_encoder():
        # Build an RNN

        dropout_lstm_cell1 = tf.nn.rnn_cell.DropoutWrapper(
                        cell=tf.nn.rnn_cell.LSTMCell(self.num_units, state_is_tuple=True),
                        input_keep_prob=1.0,
                        output_keep_prob=1.0,
                        state_keep_prob=1.0)


        self.encoder_lstm_cells = [dropout_lstm_cell1, tf.nn.rnn_cell.LSTMCell(self.num_units, state_is_tuple=True)]

        self.stacked_encocder_cell = tf.nn.rnn_cell.MultiRNNCell(self.encoder_lstm_cells)

        # Build a dynamic RNN
        #   encoder_outputs: [batch_size, max_time, num_units]
        #   encoder_state:   [batch_size, num_units]  -> final state
        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                                            self.stacked_encocder_cell, self.src_embedded,
                                            sequence_length=self.src_sentence_lengths,
                                            initial_state=self.stacked_encocder_cell.zero_state(s[0],dtype=tf.float32),
                                            time_major=False)

    ################### def build_decoder():
        # Build an RNN Cell
        dropout_lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(
                        cell=tf.nn.rnn_cell.LSTMCell(self.num_units, state_is_tuple=True),
                        input_keep_prob=1.0,
                        output_keep_prob=1.0,
                        state_keep_prob=1.0)

        self.decoder_lstm_cells = [dropout_lstm_cell2, tf.nn.rnn_cell.LSTMCell(self.num_units, state_is_tuple=True)]

        self.stacked_decoder_cell = tf.nn.rnn_cell.MultiRNNCell(self.decoder_lstm_cells)



        # Helper - A helper for use during training. Only reads inputs.
        #          Returned sample_ids are the argmax of the RNN output logits.
        self.train_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=self.tgt_embedded,
                    sequence_length=[self.max_sentence_length]*self.batch_size,
                    time_major=False)

        # Decoder
        self.projection_layer = tf.layers.Dense(self.params['vocab_tgt_size'], use_bias=True)

        self.train_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=self.stacked_decoder_cell, helper=self.train_helper,
                    initial_state=self.encoder_state,
                    output_layer=self.projection_layer)

    ################## decoding
        # Dynamic decoding
        # (final_outputs, final_state, final_sequence_lengths)
        (self.outputs, _ , _ ) = tf.contrib.seq2seq.dynamic_decode(
                    self.train_decoder, output_time_major=False, impute_finished=True)
        self.logits = self.outputs.rnn_output

    ################## calculating Loss

        # this function applies softmax internally
        self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.tgt_word_ids, logits=self.logits)

        # the denominator could also be batch_size * num_time_steps
        self.target_weights = tf.sequence_mask(lengths=self.tgt_sentence_lengths,
                                            maxlen=self.max_sentence_length,
                                            dtype=tf.float32)

        self.train_loss = (tf.reduce_sum(self.crossent * self.target_weights) / self.batch_size)

    ################## gradient and optimisation
        # backpropagation
        self.trainable_params = tf.trainable_variables() # return a list of Variable objects
        self.gradients = tf.gradients(self.train_loss, self.trainable_params)
        max_gradient_norm = 1.0 # set to value like 1.0, 5.0
        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, max_gradient_norm)

        # optimisation
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.trainable_params))

    # ------------------------ inference ------------------------ #
        # Inference Helper
        self.infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=self.embedding_decoder,
                start_tokens=tf.fill([s[0]], self.params['go_id']),
                end_token=self.params['eos_id'])
        # self.infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        #         embedding=self.tgt_word_embeddings,
        #         start_tokens=tf.fill([s[0]], self.params['sos_id']),
        #         end_token=self.params['eos_id'])

        # Decoder
        self.infer_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.stacked_decoder_cell,
            helper=self.infer_helper,
            initial_state=self.encoder_state,
            output_layer=self.projection_layer)

        # Dynamic decoding
        (self.infer_outputs, _ , _ ) = tf.contrib.seq2seq.dynamic_decode(
                self.infer_decoder,
                maximum_iterations=self.max_sentence_length,
                output_time_major=False, impute_finished=True)

        self.translations = self.infer_outputs.sample_id

        ###############################################################
        ###############################################################
        self.infer_logits = self.infer_outputs.rnn_output
        paddings = [[0, 0], [0, self.max_sentence_length-tf.shape(self.infer_logits)[1]], [0, 0]]
        self.infer_logits = tf.pad(self.infer_logits, paddings, 'CONSTANT', constant_values=-1)


        self.infer_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.tgt_word_ids, logits=self.infer_logits)

        self.infer_target_weights = tf.sequence_mask(lengths=self.tgt_sentence_lengths,
                                            maxlen=self.max_sentence_length,
                                            dtype=tf.float32)

        self.infer_loss = (tf.reduce_sum(self.infer_crossent * self.infer_target_weights) / self.batch_size)
        ###############################################################
        ###############################################################
    # ----------------------------------------------------------- #

    # callable for infer_helper
    def embedding_decoder(self, ids):
        return tf.nn.embedding_lookup(self.tgt_word_embeddings, ids)



def train_model():
    batches, vocab_size, src_word2id, tgt_word2id = construct_training_data_batches()

    first_batch = batches[0]

    # tgt_id2word = ['<emp>'] + list(tgt_word2id.keys())
    tgt_id2word = list(tgt_word2id.keys())

    params = {'vocab_src_size': vocab_size['src'],
            'vocab_tgt_size': vocab_size['tgt'],
            'go_id':  tgt_word2id['<go>'],
            'eos_id':  tgt_word2id['</s>']}

    model = EncoderDecoder(params)
    model.build_network()

    use_gpu = True

    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        sess_config = tf.ConfigProto()
    else:
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True # Whether the GPU memory usage can grow dynamically.
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.95 # The fraction of GPU memory that the process can use.

    # sess = tf.Session(config=sess_config) # sess.close() / sess.run()
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        tf_variables = tf.trainable_variables()
        for i in range(len(tf_variables)):
            print(tf_variables[i])

        for epoch in range(40):
            print("num_batches = ", len(batches))
            for i, batch in enumerate(batches):

                feed_dict = { model.src_word_ids: batch['src_word_ids'],
                            model.tgt_word_ids: batch['tgt_word_ids'],
                            model.src_sentence_lengths: batch['src_sentence_lengths'],
                            model.tgt_sentence_lengths: batch['tgt_sentence_lengths']}

                # pdb.set_trace()
                train_logits, train_src_word_ids, my_embedding, train_src_embedded, train_enc_state, train_loss, _ = sess.run([model.logits, model.src_word_ids, model.src_word_embeddings, model.src_embedded, model.encoder_state, model.train_loss, model.train_op], feed_dict=feed_dict)
                # sentence 4 #
                logits4 = train_logits[4]
                sentence4_out = [tgt_id2word[id] for id in np.argmax(logits4, axis=1)]
                sentence4_in  = [tgt_id2word[id] for id in train_src_word_ids[4]]
                # pdb.set_trace()
                # print(' '.join(sentence4))


                # [aa] = sess.run([model.s], feed_dict=feed_dict)
                # pdb.set_trace()
                if i % 10 == 0:
                    # print("batch: {} --- train_loss: {}".format(i, train_loss))

                    # --------------------------------------------------- #

                    infer_dict = { model.src_word_ids: batch['src_word_ids'],
                                model.tgt_word_ids: batch['tgt_word_ids'],
                                model.src_sentence_lengths: batch['src_sentence_lengths'],
                                model.tgt_sentence_lengths: batch['tgt_sentence_lengths']}

                    [inf_src_embedded, inf_enc_state, my_translations, infer_loss] = sess.run([model.src_embedded, model.encoder_state, model.translations, model.infer_loss], feed_dict=infer_dict)

                    print("batch: {} --- train_loss: {:.5f} | inf_loss: {:.5f}".format(i, train_loss, infer_loss))

                    # --------------------------------------------------- #

                if i % 50 == 0:
                    # --------------------------------------------------- #
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
                    my_sent_ids = [ids + [src_word2id['</s>']]*(32-len(ids)) for ids in my_sent_ids]


                    infer_dict = {model.src_word_ids: my_sent_ids,
                                model.src_sentence_lengths: my_sent_len}

                    [my_translations] = sess.run([model.translations], feed_dict=infer_dict)
                    # pdb.set_trace()
                    for my_sent in my_translations:
                        my_words = [tgt_id2word[id] for id in my_sent]
                        print(' '.join(my_words))
                    # --------------------------------------------------- #

            print("########## EPOCH {} done ##########".format(epoch))




def main():
    train_model()

if __name__ == '__main__':
    main()
