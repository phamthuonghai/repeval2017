"""

Functions and components that can be slotted into tensorflow models.

TODO: Write functions for various types of attention.

"""

import tensorflow as tf


def length(sequence):
    """
    Get true length of sequences (without padding), and mask for true-length in max-length.

    Input of shape: (batch_size, max_seq_length, hidden_dim)
    Output shapes, 
    length: (batch_size)
    mask: (batch_size, max_seq_length, 1)
    """
    populated = tf.sign(tf.abs(sequence))
    length = tf.cast(tf.reduce_sum(populated, axis=1), tf.int32)
    mask = tf.cast(tf.expand_dims(populated, -1), tf.float32)
    return length, mask


def biLSTM(inputs, dim, seq_len, name, reuse=False):
    """
    A Bi-Directional LSTM layer. Returns forward and backward hidden states as a tuple, and cell states as a tuple.

    Ouput of hidden states: [(batch_size, max_seq_length, hidden_dim), (batch_size, max_seq_length, hidden_dim)]
    Same shape for cell states.
    """
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('forward' + name, reuse=reuse):
            lstm_fwd = tf.contrib.rnn.LSTMCell(num_units=dim)
        with tf.variable_scope('backward' + name, reuse=reuse):
            lstm_bwd = tf.contrib.rnn.LSTMCell(num_units=dim)

        hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fwd, cell_bw=lstm_bwd, inputs=inputs,
                                                                     sequence_length=seq_len, dtype=tf.float32)

    return hidden_states, cell_states


def LSTM(inputs, dim, seq_len, name):
    """
    An LSTM layer. Returns hidden states and cell states as a tuple.

    Ouput shape of hidden states: (batch_size, max_seq_length, hidden_dim)
    Same shape for cell states.
    """
    with tf.name_scope(name):
        cell = tf.contrib.rnn.LSTMCell(num_units=dim)
        hidden_states, cell_states = tf.nn.dynamic_rnn(cell, inputs=inputs,
                                                       sequence_length=seq_len, dtype=tf.float32, scope=name)

    return hidden_states, cell_states


def last_output(output, true_length):
    """
    To get the last hidden layer form a dynamically unrolled RNN.
    Input of shape (batch_size, max_seq_length, hidden_dim).

    true_length: Tensor of shape (batch_size). Such a tensor is given by the length() function.
    Output of shape (batch_size, hidden_dim).
    """
    max_length = int(output.get_shape()[1])
    length_mask = tf.expand_dims(tf.one_hot(true_length-1, max_length, on_value=1., off_value=0.), -1)
    vlast_output = tf.reduce_sum(tf.multiply(output, length_mask), 1)
    return vlast_output


def masked_softmax(scores, mask):
    """
    Used to calculcate a softmax score with true sequence length (without padding), rather than max-sequence length.

    Input shape: (batch_size, max_seq_length, hidden_dim). 
    mask parameter: Tensor of shape (batch_size, max_seq_length). Such a mask is given by the length() function.
    """
    numerator = tf.exp(tf.subtract(scores, tf.reduce_max(scores, 1, keep_dims=True))) * mask
    denominator = tf.reduce_sum(numerator, 1, keep_dims=True)
    weights = tf.div(numerator, denominator)
    return weights


def self_attention(inputs, s1_dim, s2_dim, batch_size, name, reuse=False):
    """
    A self-attentive block as described in https://arxiv.org/pdf/1703.03130.pdf

    Input shape: (batch_size, max_seq_length, 2*hidden_dim).
    Output shape of m: (s2_dim, 2*hidden_dim).
    penal: a real number
    """
    with tf.variable_scope(name, reuse=reuse):
        s1 = tf.layers.dense(inputs, s1_dim, activation=tf.nn.tanh, name=name+'_s1')  # (?, max_seq_length, s1_dim)
        a = tf.layers.dense(s1, s2_dim, activation=tf.nn.softmax, name=name+'_s2')  # (?, max_seq_length, s2_dim)
        at = tf.transpose(a, perm=[0, 2, 1])
        m = tf.matmul(at, inputs)  # (?, s2_dim, 2*hidden_dim)
        penal = tf.norm(tf.subtract(tf.matmul(at, a), tf.eye(s2_dim, batch_shape=[batch_size])), ord='fro', axis=(1, 2))
        penal = tf.reduce_mean(tf.square(penal))
    return tf.reshape(m, [batch_size, -1]), penal
