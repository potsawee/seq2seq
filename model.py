import tensorflow as tf
'''
Encoder-Decoder architecture
'''

def load_data():


def construct_network():
################### def build_encoder():
    # Build an RNN Cell
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    # Build a dynamic RNN
    #   encoder_outputs: [max_time, batch_size, num_units]
    #   encoder_state: [batch_size, num_units]  -> final state
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                                        encoder_cell, encoder_emb_inp,
                                        sequence_length=source_sequence_length,
                                        time_major=True)

################### def build_decoder():
    # Build an RNN Cell
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    # Helper - A helper for use during training. Only reads inputs.
    #          Returned sample_ids are the argmax of the RNN output logits.
    helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_emb_inp, decoder_lengths, time_major=True)

    # Decoder
    projection_layer = tf.layers.Dense(tgt_vocab_size, use_bias=True)
    decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, helper, encoder_state,
                output_layer=projection_layer)

    encoder_outputs, enocoder_state = build_encoder()
    decoder = build_decoder()

################## decoding
    # Dynamic decoding
    outputs, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, output_time_major=True)
    logits = outputs.rnn_output

################## calculating Loss

    # this function applies softmax internally
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=decoder_outputs, logits=logits)

    # the denominator could also be batch_size * num_time_steps
    train_loss = (tf.reduce_sum(crossent * target_weights) / batch_size)

################## gradient and optimisation
    # backpropagation
    params = tf.trainable_variable() # return a list of Variable objects
    gradients = tf.gradients(train_loss, params)
    max_gradient_norm = 1.0 # set to value like 1.0, 5.0
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

    # optimisation
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_step = optimizer.apply_gradients(zip(clipped_gradients, params))


def dev():




if __name__ == '__main__':
    dev()
