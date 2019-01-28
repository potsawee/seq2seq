import collections
import math
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
        self.beam_width = config['beam_width']
        self._residual = config['residual']
        self._scheduled_sampling = config['scheduled_sampling']
        self._counter = 0 # for counting the number of iterations
        self._train_epochs = config['num_epochs']
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
        tgt_word_ids_with_go = tf.concat( [tf.fill([self.batch_size, 1], go_id), self.tgt_word_ids], 1)

        ############################## embeddings ##############################
        self.src_word_embeddings = tf.get_variable("src_word_embeddings",
                                shape=[self.params['vocab_src_size'], self.embedding_size],
                                initializer=tf.glorot_normal_initializer())
        self.tgt_word_embeddings = tf.get_variable("tgt_word_embeddings",
                                shape=[self.params['vocab_tgt_size'], self.embedding_size],
                                initializer=tf.glorot_normal_initializer())


        src_embedded = tf.nn.embedding_lookup(self.src_word_embeddings, self.src_word_ids)
        tgt_embedded = tf.nn.embedding_lookup(self.tgt_word_embeddings, tgt_word_ids_with_go)
        # Look up embedding:
        #   encoder_inputs:  [batch_size, max_time]
        #   encoder_emb_inp: [batch_size, max_time, embedding_size]

        s = tf.shape(self.src_word_ids) # s[0] = batch_size , s[1] = max_sentecce_length

        ############################## Encoder ##############################
        with tf.variable_scope("encoder"):
            # For bi-directional model the encoder effectively has double layers
            assert (self.num_layers % 2 == 0), "num_layers must be even"
            num_bi_encoder_layers = int(self.num_layers / 2)
            # ----- forward ----- #
            with tf.variable_scope("enc_forward"):
                cell_list = []
                for i in range(num_bi_encoder_layers):
                    single_cell = self.build_single_cell(self.num_units, self.dropout, self._residual)
                    cell_list.append(single_cell)

            if num_bi_encoder_layers == 1:
                fw_encoder_cell = cell_list[0]
            elif num_bi_encoder_layers > 1:
                fw_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
            else:
                raise ValueError('num_layers error')
            # ----- backward ----- #
            with tf.variable_scope("enc_backward"):
                cell_list = []
                for i in range(num_bi_encoder_layers):
                    single_cell = self.build_single_cell(self.num_units, self.dropout, self._residual)
                    cell_list.append(single_cell)

            if num_bi_encoder_layers == 1:
                bw_encoder_cell = cell_list[0]
            elif num_bi_encoder_layers > 1:
                bw_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
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
            bi_encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                    fw_encoder_cell, bw_encoder_cell,
                    src_embedded,
                    sequence_length=self.src_sentence_lengths,
                    initial_state_fw=fw_encoder_cell.zero_state(s[0],dtype=tf.float32),
                    initial_state_bw=bw_encoder_cell.zero_state(s[0],dtype=tf.float32),
                    time_major=False)

            encoder_outputs = tf.concat(bi_encoder_outputs, -1)

            if num_bi_encoder_layers == 1:
                encoder_state = bi_encoder_state
            elif num_bi_encoder_layers > 1:
                # alternatively concat forward and backward states
                encoder_state = []
                for i in range(num_bi_encoder_layers):
                    encoder_state.append(bi_encoder_state[0][i])  # forward
                    encoder_state.append(bi_encoder_state[1][i])  # backward
                encoder_state = tuple(encoder_state)
            else:
                raise ValueError('num_layers error')



        ############################## Decoder ##############################
        with tf.variable_scope("decoder"):
            cell_list = []

            # top of the stack -> no residual
            single_cell = self.build_single_cell(self.num_units, self.dropout, residual=False)
            cell_list.append(single_cell)

            for i in range(self.num_layers-1):
                single_cell = self.build_single_cell(self.num_units, self.dropout, self._residual)
                cell_list.append(single_cell)


            if self.num_layers == 1:
                stacked_decoder_cell = cell_list[0]
            elif self.num_layers > 1:
                stacked_decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
            else:
                raise ValueError('num_layers error')

            # ------------------------- Training --------------------------- #

            # note that previously encoder_outputs is the set of all source 'hidden' states at the top layer
            with tf.variable_scope('shared_attention_mechanism'):
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    num_units=self.num_units,
                    memory=encoder_outputs,
                    memory_sequence_length=self.src_sentence_lengths)

            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=stacked_decoder_cell,
                attention_mechanism=attention_mechanism,
                attention_layer_size=self.num_units)

            # Helper - A helper for use during training. Only reads inputs.
            #          Returned sample_ids are the argmax of the RNN output logits.

            if not self._scheduled_sampling:
                # scheduled_sampling for training is disabled
                print('scheduled sampling disabled')
                train_helper = tf.contrib.seq2seq.TrainingHelper(
                            inputs=tgt_embedded,
                            sequence_length=[self.max_sentence_length]*self.batch_size,
                            time_major=False)

            else:
                # scheduled_sampling for training is enabled
                # sampling_probability (if 0.0 means no sampling, 1.0 means always sampling)
                print('scheduled sampling enabled')
                train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                            inputs=tgt_embedded,
                            sequence_length=[self.max_sentence_length]*self.batch_size,
                            embedding=self.embedding_decoder,
                            sampling_probability=1.0*self._counter/self._train_epochs, # linear increase
                            time_major=False)

            projection_layer = tf.layers.Dense(self.params['vocab_tgt_size'], use_bias=True)

            train_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=decoder_cell, helper=train_helper,
                        initial_state=decoder_cell.zero_state(s[0],dtype=tf.float32).clone(cell_state=encoder_state),
                        output_layer=projection_layer)


            # Dynamic decoding
            # (final_outputs, final_state, final_sequence_lengths)
            with tf.variable_scope('decode_with_shared_attention'):
                (outputs, _ , _ ) = tf.contrib.seq2seq.dynamic_decode(
                            train_decoder, output_time_major=False, impute_finished=True)
            logits = outputs.rnn_output

            # -------------------- Inference -------------------- #

            # Inference Helper (1) greedy search (2) sample (3) modified-sample (4) beam search
            if self.decoding_method != 'beamsearch':
                if self.decoding_method == 'greedy':
                    infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        embedding=self.embedding_decoder,
                        start_tokens=tf.fill([s[0]], self.params['go_id']),
                        end_token=self.params['eos_id'])

                elif self.decoding_method == 'sample1':
                    infer_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                        embedding=self.embedding_decoder,
                        start_tokens=tf.fill([s[0]], self.params['go_id']),
                        end_token=self.params['eos_id'],
                        softmax_temperature=1.0)

                elif self.decoding_method == 'sample2':
                    """sample to get output & argmax to get element for the next time step"""
                    infer_helper = ModifiedSampleEmbeddingHelper(
                        embedding=self.embedding_decoder,
                        start_tokens=tf.fill([s[0]], self.params['go_id']),
                        end_token=self.params['eos_id'],
                        softmax_temperature=1.0)

                infer_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell,
                    helper=infer_helper,
                    initial_state=decoder_cell.zero_state(s[0],dtype=tf.float32).clone(cell_state=encoder_state),
                    output_layer=projection_layer)

            elif self.decoding_method == 'beamsearch':

                encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_width)
                src_sentence_lengths_beam = tf.contrib.seq2seq.tile_batch(self.src_sentence_lengths, multiplier=self.beam_width)
                encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=self.beam_width)

                decoder_initial_state=decoder_cell.zero_state(s[0]*self.beam_width, dtype=tf.float32).clone(cell_state=encoder_state)

                with tf.variable_scope('shared_attention_mechanism', reuse=True):
                    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                        num_units=self.num_units,
                        memory=encoder_outputs,
                        memory_sequence_length=src_sentence_lengths_beam)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell=stacked_decoder_cell,
                    attention_mechanism=attention_mechanism,
                    attention_layer_size=self.num_units)

                infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=self.embedding_decoder,
                    start_tokens=tf.fill([s[0]], self.params['go_id']),
                    end_token=self.params['eos_id'],
                    initial_state=decoder_initial_state,
                    beam_width=self.beam_width,
                    output_layer=projection_layer,
                    length_penalty_weight=0.0)

            else:
                raise ValueError('decoding method error: only GreedySearch or BeamSearch')


            # Dynamic decoding
            if self.decoding_method != 'beamsearch':
                with tf.variable_scope('decode_with_shared_attention', reuse=True):
                    (infer_outputs, _ , _ ) = tf.contrib.seq2seq.dynamic_decode(
                            infer_decoder,
                            maximum_iterations=self.max_sentence_length,
                            output_time_major=False, impute_finished=True)

                self.translations = infer_outputs.sample_id # shape = [batch_size, max_sentence_length]


            elif self.decoding_method == 'beamsearch':
                with tf.variable_scope('decode_with_shared_attention', reuse=True):
                    (infer_outputs, _ , _ ) = tf.contrib.seq2seq.dynamic_decode(
                            infer_decoder,
                            maximum_iterations=self.max_sentence_length,
                            output_time_major=False, impute_finished=False)

                # outputs: predicted_ids, beam_search_decoder_output
                self.predicted_ids = infer_outputs.predicted_ids # shape = [batch_size, max_sentence_length, beam_width]
                self.translations = self.predicted_ids[:,:,0] # the first one has the highest probability


        ############################## Calculating Loss ##############################
        # -------------------- Training -------------------- #
        # this function applies softmax internally
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.tgt_word_ids, logits=logits)

        # the denominator could also be batch_size * num_time_steps
        target_weights = tf.sequence_mask(lengths=self.tgt_sentence_lengths,
                                            maxlen=self.max_sentence_length,
                                            dtype=tf.float32)

        self.train_loss = (tf.reduce_sum(crossent * target_weights) / self.batch_size)

        # -------------------- Inference -------------------- #
        if self.decoding_method != 'beamsearch':
            infer_logits = infer_outputs.rnn_output
            infer_paddings = [[0, 0], [0, self.max_sentence_length-tf.shape(infer_logits)[1]], [0, 0]]
            infer_logits = tf.pad(infer_logits, infer_paddings, 'CONSTANT', constant_values=-1)

            infer_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=self.tgt_word_ids, logits=infer_logits)

            infer_target_weights = tf.sequence_mask(lengths=self.tgt_sentence_lengths,
                                                maxlen=self.max_sentence_length,
                                                dtype=tf.float32)

            self.infer_loss = (tf.reduce_sum(infer_crossent * infer_target_weights) / self.batch_size)

        ############################## Gradient and Optimisation ##############################
        # backpropagation
        trainable_params = tf.trainable_variables() # return a list of Variable objects
        gradients = tf.gradients(self.train_loss, trainable_params)
        max_gradient_norm = 1.0 # set to value like 1.0, 5.0
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

        # optimisation
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, trainable_params))

    # end build_network()

    def adapt_weights(self, param_names):
        '''
        Args:
            - param_names: a list of names (strings) of the weights/biases to be adapted
        '''

        # Create variable scope for the trainable parts of the graph: tf.variable_scope('train').
        # get trainable variables
        all_trainable_vars = tf.trainable_variables()
        adapt_vars = []
        for var in all_trainable_vars:
            if var.name in param_names:
                adapt_vars.append(var)
        # train only the variables of a particular scope
        gradients = tf.gradients(self.train_loss, adapt_vars)
        max_gradient_norm = 1.0 # set to value like 1.0, 5.0
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.adapt_op = self.optimizer.apply_gradients(zip(clipped_gradients, adapt_vars))

    # callable for infer_helper
    def embedding_decoder(self, ids):
        return tf.nn.embedding_lookup(self.tgt_word_embeddings, ids)

    # methods for build the network
    def build_single_cell(self, num_units, dropout, residual=False):
        '''build a single cell'''
        # LSTM cell
        single_cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)
        single_cell = tf.nn.rnn_cell.DropoutWrapper(cell=single_cell, input_keep_prob=1.0-dropout)

        if residual:
            single_cell = tf.nn.rnn_cell.ResidualWrapper(single_cell)
            print('build RNN cell with residual connection')

        return single_cell

    def increment_counter(self):
        '''increment counter by 1'''
        self._counter += 1
