import tensorflow as tf
from utils import blocks


class MyModel(object):
    def __init__(self, **kwargs):
        # Define hyperparameters
        self.embedding_dim = kwargs['word_embedding_dim']
        self.dim = kwargs['hidden_embedding_dim']
        self.sequence_length = kwargs['seq_length']
        self.batch_size = kwargs['batch_size']
        self.r = kwargs['s2_dim']

        # Define the placeholders
        self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.y = tf.placeholder(tf.int32, [None])
        self.keep_rate_ph = tf.placeholder(tf.float32, [])
        self.prem_dep = tf.placeholder(tf.float32, [None, self.r, self.sequence_length])
        self.hypo_dep = tf.placeholder(tf.float32, [None, self.r, self.sequence_length])

        # Define parameters
        self.E = tf.Variable(kwargs['embeddings'], trainable=kwargs['emb_train'])

        self.W_mlp = tf.Variable(tf.random_normal([self.dim * 8 * self.r, self.dim], stddev=0.1))
        self.b_mlp = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

        self.W_cl = tf.Variable(tf.random_normal([self.dim, 3], stddev=0.1))
        self.b_cl = tf.Variable(tf.random_normal([3], stddev=0.1))

        # Function for embedding lookup and dropout at embedding layer
        def emb_drop(x):
            emb = tf.nn.embedding_lookup(self.E, x)
            emb_drop = tf.nn.dropout(emb, self.keep_rate_ph)
            return emb_drop

        # Get lengths of unpadded sentences
        prem_seq_lengths, prem_mask = blocks.length(self.premise_x)
        hyp_seq_lengths, hyp_mask = blocks.length(self.hypothesis_x)

        # BiLSTM layer
        premise_in = emb_drop(self.premise_x)
        hypothesis_in = emb_drop(self.hypothesis_x)

        premise_outs, c1 = blocks.biLSTM(premise_in, dim=self.dim, seq_len=prem_seq_lengths, name='premise')
        if kwargs['shared_encoder']:
            hypothesis_outs, c2 = blocks.biLSTM(hypothesis_in, dim=self.dim, seq_len=hyp_seq_lengths,
                                                name='premise', reuse=True)
        else:
            hypothesis_outs, c2 = blocks.biLSTM(hypothesis_in, dim=self.dim, seq_len=hyp_seq_lengths, name='hypothesis')

        premise_bi = tf.concat(premise_outs, axis=2)
        hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

        premise_ave = blocks.dep_attention(premise_bi, self.prem_dep, batch_size=self.batch_size)
        hypothesis_ave = blocks.dep_attention(hypothesis_bi, self.hypo_dep, batch_size=self.batch_size)

        # Mou et al. concat layer ###
        diff = tf.subtract(premise_ave, hypothesis_ave)
        mul = tf.multiply(premise_ave, hypothesis_ave)
        h = tf.concat([premise_ave, hypothesis_ave, diff, mul], 1)

        self.features = h

        # MLP layer
        h_mlp = tf.nn.relu(tf.matmul(h, self.W_mlp) + self.b_mlp)
        # Dropout applied to classifier
        h_drop = tf.nn.dropout(h_mlp, self.keep_rate_ph)

        # Get prediction
        self.logits = tf.matmul(h_drop, self.W_cl) + self.b_cl

        # Define the cost function
        self.total_cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
