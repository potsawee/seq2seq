import collections
import tensorflow as tf
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
        self.num_units = config['num_units']
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
        # Build an RNN

        dropout_lstm_cell1 = tf.nn.rnn_cell.DropoutWrapper(
                        cell=tf.nn.rnn_cell.LSTMCell(self.num_units, state_is_tuple=True),
                        input_keep_prob=1.0,
                        output_keep_prob=1.0,
                        state_keep_prob=1.0)


        encoder_lstm_cells = [dropout_lstm_cell1, tf.nn.rnn_cell.LSTMCell(self.num_units, state_is_tuple=True)]

        self.stacked_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_lstm_cells)

        # Build a dynamic RNN
        #   encoder_outputs: [batch_size, max_time, num_units]
        #   encoder_state:   [batch_size, num_units]  -> final state
        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                                            self.stacked_encoder_cell, self.src_embedded,
                                            sequence_length=self.src_sentence_lengths,
                                            initial_state=self.stacked_encoder_cell.zero_state(s[0],dtype=tf.float32),
                                            time_major=False)

        ############################## Decoder ##############################
        # Build an RNN Cell
        dropout_lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(
                        cell=tf.nn.rnn_cell.LSTMCell(self.num_units, state_is_tuple=True),
                        input_keep_prob=1.0,
                        output_keep_prob=1.0,
                        state_keep_prob=1.0)

        decoder_lstm_cells = [dropout_lstm_cell2, tf.nn.rnn_cell.LSTMCell(self.num_units, state_is_tuple=True)]

        self.stacked_decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_lstm_cells)

        # -------------------- Training -------------------- #

        # Helper - A helper for use during training. Only reads inputs.
        #          Returned sample_ids are the argmax of the RNN output logits.
        self.train_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=self.tgt_embedded,
                    sequence_length=[self.max_sentence_length]*self.batch_size,
                    time_major=False)

        self.projection_layer = tf.layers.Dense(self.params['vocab_tgt_size'], use_bias=True)

        self.train_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=self.stacked_decoder_cell, helper=self.train_helper,
                    initial_state=self.encoder_state,
                    output_layer=self.projection_layer)

        # Dynamic decoding
        # (final_outputs, final_state, final_sequence_lengths)
        (self.outputs, _ , _ ) = tf.contrib.seq2seq.dynamic_decode(
                    self.train_decoder, output_time_major=False, impute_finished=True)
        self.logits = self.outputs.rnn_output

        # -------------------- Inference -------------------- #
        # Inference Helper
        self.infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=self.embedding_decoder,
                start_tokens=tf.fill([s[0]], self.params['go_id']),
                end_token=self.params['eos_id'])

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
