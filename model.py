import collections
import tensorflow as tf
from helper import ModifiedSampleEmbeddingHelper
import pdb

'''
Encoder-Decoder architecture
'''

class EncoderDecoder(object):
    def __init__(self, config, params):
        # hyper-parameters / configurations
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.embedding_size = config['embedding_size']
        self.num_layers = config['num_layers']
        self.num_units = config['num_units']
        self.decoding_method = config['decoding_method']
        self.max_sentence_length = config['max_sentence_length']
        self.params = params


    def build_network(self):
        '''
        Build Encoder Decoder Network
            - Encoder
            - Decoder (Train & Inference)
        '''
        ############################## placeholders ##############################
        self.src_word_ids = tf.placeholder(tf.int32, [None, None], name="src_word_ids")
        self.tgt_word_ids = tf.placeholder(tf.int32, [None, None], name="tgt_word_ids")
        self.src_sentence_lengths = tf.placeholder(tf.int32, [None], name="src_sentence_lengths")
        self.tgt_sentence_lengths = tf.placeholder(tf.int32, [None], name="tgt_sentence_lengths")

        self.dropout = tf.placeholder(tf.float32, name="dropout")

        ############################## padding <go> ##############################
        go_id = self.params['go_id']
        self.tgt_word_ids_with_go = tf.concat( [tf.fill([self.batch_size, 1], go_id), self.tgt_word_ids], 1)

        ############################## embeddings ##############################
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

        ############################## Encoder ##############################
        # For bi-directional model the encoder effectively has double layers
        assert (self.num_layers % 2 == 0), "num_layers must be even"
        num_bi_encoder_layers = int(self.num_layers / 2)
        # ----- forward ----- #
        cell_list = []
        for i in range(num_bi_encoder_layers):
            single_cell = self.build_single_cell(self.num_units, self.dropout)
            cell_list.append(single_cell)

        if num_bi_encoder_layers == 1:
            self.fw_encoder_cell = cell_list[0]
        elif num_bi_encoder_layers > 1:
            self.fw_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
        else:
            raise ValueError('num_layers error')
        # ----- backward ----- #
        cell_list = []
        for i in range(num_bi_encoder_layers):
            single_cell = self.build_single_cell(self.num_units, self.dropout)
            cell_list.append(single_cell)

        if num_bi_encoder_layers == 1:
            self.bw_encoder_cell = cell_list[0]
        elif num_bi_encoder_layers > 1:
            self.bw_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
        else:
            raise ValueError('num_layers error')

        # Build a dynamic RNN
        #   encoder_outputs: [batch_size, max_time, num_units]
        #   encoder_state:   [batch_size, num_units]  -> final state
        # self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
        #                                     self.encoder_cell, self.src_embedded,
        #                                     sequence_length=self.src_sentence_lengths,
        #                                     initial_state=self.encoder_cell.zero_state(s[0],dtype=tf.float32),
        #                                     time_major=False)

        # Bi-directional RNN
        self.bi_encoder_outputs, self.bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                self.fw_encoder_cell, self.bw_encoder_cell,
                self.src_embedded,
                sequence_length=self.src_sentence_lengths,
                initial_state_fw=self.fw_encoder_cell.zero_state(s[0],dtype=tf.float32),
                initial_state_bw=self.bw_encoder_cell.zero_state(s[0],dtype=tf.float32),
                time_major=False)

        self.encoder_outputs = tf.concat(self.bi_encoder_outputs, -1)

        if num_bi_encoder_layers == 1:
            self.encoder_state = self.bi_encoder_state
        elif num_bi_encoder_layers > 1:
            # alternatively concat forward and backward states
            encoder_state = []
            for i in range(num_bi_encoder_layers):
                encoder_state.append(self.bi_encoder_state[0][i])  # forward
                encoder_state.append(self.bi_encoder_state[1][i])  # backward
            self.encoder_state = tuple(encoder_state)
        else:
            raise ValueError('num_layers error')



        ############################## Decoder ##############################
        cell_list = []
        for i in range(self.num_layers):
            single_cell = self.build_single_cell(self.num_units, self.dropout)
            cell_list.append(single_cell)

        if self.num_layers == 1:
            self.decoder_cell = cell_list[0]
        elif self.num_layers > 1:
            self.decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
        else:
            raise ValueError('num_layers error')

        # --------------- Attention Mechanism --------------- #
        # note that previously encoder_outputs is the set of all source 'hidden' states at the top layer
        self.attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=self.num_units,
            memory=self.encoder_outputs,
            memory_sequence_length=self.src_sentence_lengths)

        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=self.decoder_cell,
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.num_units)

        # -------------------- Training -------------------- #

        # Helper - A helper for use during training. Only reads inputs.
        #          Returned sample_ids are the argmax of the RNN output logits.
        self.train_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=self.tgt_embedded,
                    sequence_length=[self.max_sentence_length]*self.batch_size,
                    time_major=False)

        self.projection_layer = tf.layers.Dense(self.params['vocab_tgt_size'], use_bias=True)

        self.train_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=self.decoder_cell, helper=self.train_helper,
                    initial_state=self.decoder_cell.zero_state(s[0],dtype=tf.float32),
                    output_layer=self.projection_layer)

        # Dynamic decoding
        # (final_outputs, final_state, final_sequence_lengths)
        (self.outputs, _ , _ ) = tf.contrib.seq2seq.dynamic_decode(
                    self.train_decoder, output_time_major=False, impute_finished=True)
        self.logits = self.outputs.rnn_output

        # -------------------- Inference -------------------- #
        # Inference Helper (1) greedy search (2) sample (3) modified-sample (4) beam search
        if self.decoding_method in ['greedy', 'sample1', 'sample2']:
            if self.decoding_method == 'greedy':
                self.infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=self.embedding_decoder,
                    start_tokens=tf.fill([s[0]], self.params['go_id']),
                    end_token=self.params['eos_id'])

            elif self.decoding_method == 'sample1':
                self.infer_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                    embedding=self.embedding_decoder,
                    start_tokens=tf.fill([s[0]], self.params['go_id']),
                    end_token=self.params['eos_id'],
                    softmax_temperature=1.0)

            elif self.decoding_method == 'sample2':
                """sample to get output & argmax to get element for the next time step"""
                self.infer_helper = ModifiedSampleEmbeddingHelper(
                    embedding=self.embedding_decoder,
                    start_tokens=tf.fill([s[0]], self.params['go_id']),
                    end_token=self.params['eos_id'],
                    softmax_temperature=1.0)

            self.infer_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=self.infer_helper,
                initial_state=self.decoder_cell.zero_state(s[0],dtype=tf.float32),
                output_layer=self.projection_layer)

        # elif self.decoding_method == 'beam':
            # beam_width = 10
            # # Replicate encoder infos beam_width times
            # decoder_initial_state = tf.contrib.seq2seq.tile_batch(
            #     self.decoder_cell.zero_state(s[0],dtype=tf.float32),
            #     multiplier=beam_width) # set beam width = 10
            #
            # # Define a beam-search decoder
            # self.infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            #     cell=self.decoder_cell,
            #     embedding=self.embedding_decoder,
            #     start_tokens=tf.fill([s[0]], self.params['go_id']),
            #     end_token=self.params['eos_id'],
            #     initial_state=decoder_initial_state,
            #     beam_width=beam_width,
            #     output_layer=self.projection_layer,
            #     length_penalty_weight=0.0)

        else:
            raise ValueError('decoding method error: only GreedySearch or BeamSearch')


        # Dynamic decoding
        (self.infer_outputs, _ , _ ) = tf.contrib.seq2seq.dynamic_decode(
                self.infer_decoder,
                maximum_iterations=self.max_sentence_length,
                output_time_major=False, impute_finished=True)

        self.translations = self.infer_outputs.sample_id

        # 0 time
        # 1 batch_size
        # 2 beam_width
        # if self.decoding_method == 'beam':
        #     self.translations

        ############################## Calculating Loss ##############################
        # -------------------- Training -------------------- #
        # this function applies softmax internally
        self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.tgt_word_ids, logits=self.logits)

        # the denominator could also be batch_size * num_time_steps
        self.target_weights = tf.sequence_mask(lengths=self.tgt_sentence_lengths,
                                            maxlen=self.max_sentence_length,
                                            dtype=tf.float32)

        self.train_loss = (tf.reduce_sum(self.crossent * self.target_weights) / self.batch_size)

        # -------------------- Inference -------------------- #
        self.infer_logits = self.infer_outputs.rnn_output
        infer_paddings = [[0, 0], [0, self.max_sentence_length-tf.shape(self.infer_logits)[1]], [0, 0]]
        self.infer_logits = tf.pad(self.infer_logits, infer_paddings, 'CONSTANT', constant_values=-1)

        self.infer_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.tgt_word_ids, logits=self.infer_logits)

        self.infer_target_weights = tf.sequence_mask(lengths=self.tgt_sentence_lengths,
                                            maxlen=self.max_sentence_length,
                                            dtype=tf.float32)

        self.infer_loss = (tf.reduce_sum(self.infer_crossent * self.infer_target_weights) / self.batch_size)

        ############################## Gradient and Optimisation ##############################
        # backpropagation
        self.trainable_params = tf.trainable_variables() # return a list of Variable objects
        self.gradients = tf.gradients(self.train_loss, self.trainable_params)
        max_gradient_norm = 1.0 # set to value like 1.0, 5.0
        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, max_gradient_norm)

        # optimisation
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.trainable_params))

    # end build_network()

    # callable for infer_helper
    def embedding_decoder(self, ids):
        return tf.nn.embedding_lookup(self.tgt_word_embeddings, ids)
    def argmax_sample(self, outputs):
        # input: 'outputs'
        # output: 'sample_ids'
        sample_ids = tf.math.argmax(outputs, axis=-1, output_type=tf.int32)
        return sample_ids

    # methods for build the network
    def build_single_cell(self, num_units, dropout):
        '''build a single cell'''
        # LSTM cell
        single_cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)
        single_cell = tf.nn.rnn_cell.DropoutWrapper(cell=single_cell, input_keep_prob=1.0-dropout)
        return single_cell
