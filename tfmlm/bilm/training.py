'''
Train and test bidirectional language models.
'''

import os
import time
import json
import re
from t2tlight.models.transformer import attention_bias
from t2tlight.models.transformer import transformer_decoder
from t2tlight.utils.attention import add_timing_signal

import tensorflow as tf
import numpy as np

from tensorflow.python.ops.init_ops import glorot_uniform_initializer
import math

from .data import Vocabulary, UnicodeCharsVocabulary

DTYPE = 'float32'
DTYPE_INT = 'int64'

tf.logging.set_verbosity(tf.logging.INFO)


def print_variable_summary():
    import pprint
    variables = sorted([[v.name, v.get_shape()] for v in tf.global_variables()])
    pprint.pprint(variables)

def warmup_cosine(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return s*(x/warmup) + (1-s)*(0.5 * (1 + tf.cos(math.pi * x)))

def get_learning_rate(options, step, total_steps, hidden_size):
    schedule = options['learning_rate_schedule']
    max_learning_rate = options['max_learning_rate']
    min_learning_rate = options['min_learning_rate']
    if schedule == 'constant':
        print('Use constant learning rate %s' % options['max_learning_rate'])
        return tf.to_float(max_learning_rate)
    elif schedule == 'warmup_cosine':
        warmup_steps = options.get('warmup_steps', 8000)
        peak_value = 10.0 * (hidden_size ** -0.5) / math.sqrt(warmup_steps) * max_learning_rate
        warmup = tf.to_float(warmup_steps / total_steps)
        print('step type %s' % type(step))
        print('total_steps type %s' % type(total_steps))
        lr = tf.cast(peak_value * warmup_cosine(step / total_steps, warmup), dtype=tf.float32)
        print('Use warmup_cosine learning rate '
            'with warmup step %s and peak rate %s' % (warmup_steps, peak_value)
        )
        return tf.maximum(min_learning_rate, lr)
    else:
        raise ValueError('Unknown learning rate schedule '
                        '%s' % schedule)


class LanguageModel(object):
    '''
    A class to build the tensorflow computational graph for NLMs

    All hyperparameters and model configuration is specified in a dictionary
    of 'options'.

    is_training is a boolean used to control behavior of dropout layers
        and softmax.  Set to False for testing.

    The LSTM cell is controlled by the 'lstm' key in options
    Here is an example:

     'lstm': {
      'cell_clip': 5,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 5,
      'projection_dim': 512,
      'use_skip_connections': True},

        'projection_dim' is assumed token embedding size and LSTM output size.
        'dim' is the hidden state size.
        Set 'dim' == 'projection_dim' to skip a projection layer.
    '''
    def __init__(self, options, is_training):
        self.options = options
        self.is_training = is_training
        self.bidirectional = options.get('bidirectional', False)

        # use word or char inputs?
        self.char_inputs = 'char_cnn' in self.options

        # for the loss function
        self.share_embedding_softmax = options.get(
            'share_embedding_softmax', False)
        if self.char_inputs and self.share_embedding_softmax:
            raise ValueError("Sharing softmax and embedding weights requires "
                             "word input")

        self.sample_softmax = options.get('sample_softmax', True)

        self._build()

    def _build_word_embeddings(self):
        print('use word embedding')
        n_tokens_vocab = self.options['n_tokens_vocab']
        batch_size = self.options['batch_size']
        # LSTM options
        projection_dim = self.options['lstm']['projection_dim']

        # the input token_ids and word embeddings
        self.token_ids = tf.placeholder(DTYPE_INT,
                               shape=(batch_size, None),
                               name='token_ids')
        # the word embeddings
        
        if self.options.get('store_embedding_in_gpu', False):
            self.embedding_weights = tf.get_variable(
                "embedding", [n_tokens_vocab, projection_dim],
                dtype=DTYPE,
            )
            self.embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                self.token_ids)
        else:
            with tf.device("/cpu:0"):
                self.embedding_weights = tf.get_variable(
                    "embedding", [n_tokens_vocab, projection_dim],
                    dtype=DTYPE,
                )
                self.embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                    self.token_ids)

        # if a bidirectional LM then make placeholders for reverse
        # model and embeddings
        if self.bidirectional:
            self.token_ids_reverse = tf.placeholder(DTYPE_INT,
                               shape=(batch_size, None),
                               name='token_ids_reverse')
            if self.options.get('store_embedding_in_gpu', False):
                self.embedding_reverse = tf.nn.embedding_lookup(
                    self.embedding_weights, self.token_ids_reverse)
            else:
                with tf.device("/cpu:0"):
                    self.embedding_reverse = tf.nn.embedding_lookup(
                        self.embedding_weights, self.token_ids_reverse)
        
        if self.options.get('scale_embeddings', False):
            self.embedding = self.embedding * (projection_dim ** 0.5)
            if self.bidirectional:
                self.embedding_reverse = self.embedding_reverse * (projection_dim ** 0.5)

    def _build_word_char_embeddings(self):
        '''
        options contains key 'char_cnn': {

        'n_characters': 60,

        # includes the start / end characters
        'max_characters_per_token': 17,

        'filters': [
            [1, 32],
            [2, 32],
            [3, 64],
            [4, 128],
            [5, 256],
            [6, 512],
            [7, 512]
        ],
        'activation': 'tanh',

        # for the character embedding
        'embedding': {'dim': 16}

        # for highway layers
        # if omitted, then no highway layers
        'n_highway': 2,
        }
        '''
        batch_size = self.options['batch_size']
        projection_dim = self.options['lstm']['projection_dim']
    
        cnn_options = self.options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        max_chars = cnn_options['max_characters_per_token']
        char_embed_dim = cnn_options['embedding']['dim']
        n_chars = cnn_options['n_characters']
        if cnn_options['activation'] == 'tanh':
            activation = tf.nn.tanh
        elif cnn_options['activation'] == 'relu':
            activation = tf.nn.relu

        # the input character ids 
        self.tokens_characters = tf.placeholder(DTYPE_INT,
                                   shape=(batch_size, None, max_chars),
                                   name='tokens_characters')
        # the character embeddings
        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable(
                    "char_embed", [n_chars, char_embed_dim],
                    dtype=DTYPE,
                    initializer=tf.random_uniform_initializer(-1.0, 1.0)
            )
            # shape (batch_size, unroll_steps, max_chars, embed_dim)
            self.char_embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                    self.tokens_characters)

            if self.bidirectional:
                self.tokens_characters_reverse = tf.placeholder(DTYPE_INT,
                                   shape=(batch_size, None, max_chars),
                                   name='tokens_characters_reverse')
                self.char_embedding_reverse = tf.nn.embedding_lookup(
                    self.embedding_weights, self.tokens_characters_reverse)


        # the convolutions
        def make_convolutions(inp, reuse):
            with tf.variable_scope('CNN', reuse=reuse):
                convolutions = []
                for i, (width, num) in enumerate(filters):
                    if cnn_options['activation'] == 'relu':
                        # He initialization for ReLU activation
                        # with char embeddings init between -1 and 1
                        #w_init = tf.random_normal_initializer(
                        #    mean=0.0,
                        #    stddev=np.sqrt(2.0 / (width * char_embed_dim))
                        #)

                        # Kim et al 2015, +/- 0.05
                        w_init = tf.random_uniform_initializer(
                            minval=-0.05, maxval=0.05)
                    elif cnn_options['activation'] == 'tanh':
                        # glorot init
                        w_init = tf.random_normal_initializer(
                            mean=0.0,
                            stddev=np.sqrt(1.0 / (width * char_embed_dim))
                        )
                    w = tf.get_variable(
                        "W_cnn_%s" % i,
                        [1, width, char_embed_dim, num],
                        initializer=w_init,
                        dtype=DTYPE)
                    b = tf.get_variable(
                        "b_cnn_%s" % i, [num], dtype=DTYPE,
                        initializer=tf.constant_initializer(0.0))

                    conv = tf.nn.conv2d(
                            inp, w,
                            strides=[1, 1, 1, 1],
                            padding="VALID") + b
                    # now max pool
                    conv = tf.nn.max_pool(
                            conv, [1, 1, max_chars-width+1, 1],
                            [1, 1, 1, 1], 'VALID')

                    # activation
                    conv = activation(conv)
                    conv = tf.squeeze(conv, squeeze_dims=[2])

                    convolutions.append(conv)

            return tf.concat(convolutions, 2)

        # for first model, this is False, for others it's True
        reuse = tf.get_variable_scope().reuse
        embedding = make_convolutions(self.char_embedding, reuse)

        self.token_embedding_layers = [embedding]

        if self.bidirectional:
            # re-use the CNN weights from forward pass
            embedding_reverse = make_convolutions(
                self.char_embedding_reverse, True)

        # for highway and projection layers:
        #   reshape from (batch_size, n_tokens, dim) to
        n_highway = cnn_options.get('n_highway')
        use_highway = n_highway is not None and n_highway > 0
        use_proj = n_filters != projection_dim

        if use_highway or use_proj:
            embedding = tf.reshape(embedding, [-1, n_filters])
            if self.bidirectional:
                embedding_reverse = tf.reshape(embedding_reverse,
                    [-1, n_filters])

        # set up weights for projection
        if use_proj:
            assert n_filters > projection_dim
            with tf.variable_scope('CNN_proj') as scope:
                    W_proj_cnn = tf.get_variable(
                        "W_proj", [n_filters, projection_dim],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / n_filters)),
                        dtype=DTYPE)
                    b_proj_cnn = tf.get_variable(
                        "b_proj", [projection_dim],
                        initializer=tf.constant_initializer(0.0),
                        dtype=DTYPE)

        # apply highways layers
        def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
            carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
            transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
            return carry_gate * transform_gate + (1.0 - carry_gate) * x

        if use_highway:
            highway_dim = n_filters

            for i in range(n_highway):
                with tf.variable_scope('CNN_high_%s' % i) as scope:
                    W_carry = tf.get_variable(
                        'W_carry', [highway_dim, highway_dim],
                        # glorit init
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE)
                    b_carry = tf.get_variable(
                        'b_carry', [highway_dim],
                        initializer=tf.constant_initializer(-2.0),
                        dtype=DTYPE)
                    W_transform = tf.get_variable(
                        'W_transform', [highway_dim, highway_dim],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE)
                    b_transform = tf.get_variable(
                        'b_transform', [highway_dim],
                        initializer=tf.constant_initializer(0.0),
                        dtype=DTYPE)

                embedding = high(embedding, W_carry, b_carry,
                                 W_transform, b_transform)
                if self.bidirectional:
                    embedding_reverse = high(embedding_reverse,
                                             W_carry, b_carry,
                                             W_transform, b_transform)
                self.token_embedding_layers.append(
                    tf.reshape(embedding, 
                        [batch_size, -1, highway_dim])
                )

        # finally project down to projection dim if needed
        if use_proj:
            embedding = tf.matmul(embedding, W_proj_cnn) + b_proj_cnn
            if self.bidirectional:
                embedding_reverse = tf.matmul(embedding_reverse, W_proj_cnn) \
                    + b_proj_cnn
            self.token_embedding_layers.append(
                tf.reshape(embedding,
                        [batch_size, -1, projection_dim])
            )

        # reshape back to (batch_size, tokens, dim)
        if use_highway or use_proj:
            shp = [batch_size, -1, projection_dim]
            embedding = tf.reshape(embedding, shp)
            if self.bidirectional:
                embedding_reverse = tf.reshape(embedding_reverse, shp)
        
        if self.options.get('scale_embeddings', False):
            embedding = embedding * (projection_dim ** 0.5)
            if self.bidirectional:
                embedding_reverse = embedding_reverse * (projection_dim ** 0.5)
        # at last assign attributes for remainder of the model
        self.embedding = embedding
        if self.bidirectional:
            self.embedding_reverse = embedding_reverse

    def _build(self):
        if self.char_inputs:
            self._build_word_char_embeddings()
        else:
            self._build_word_embeddings()

        # now the LSTMs
        # these will collect the initial states for the forward
        #   (and reverse LSTMs if we are doing bidirectional)
        self.init_lstm_state = []
        self.final_lstm_state = []

        # get the LSTM inputs
        if self.bidirectional:
            lstm_inputs = [self.embedding, self.embedding_reverse]
        else:
            lstm_inputs = [self.embedding]

        # now compute the LSTM outputs
        if self.options.get("use_transformer", False):
            lstm_outputs = self._transformer_lm(lstm_inputs)
        else:
            lstm_outputs =  self._lstm_lm(lstm_inputs)
        self._build_loss(lstm_outputs)

    def _lstm_lm(self, lstm_inputs):
        batch_size = self.options['batch_size']
        # LSTM options
        lstm_dim = self.options['lstm']['dim']
        projection_dim = self.options['lstm']['projection_dim']
        n_lstm_layers = self.options['lstm'].get('n_layers', 1)
        dropout = self.options['dropout']
        keep_prob = 1.0 - dropout
        cell_clip = self.options['lstm'].get('cell_clip')
        proj_clip = self.options['lstm'].get('proj_clip')
        use_skip_connections = self.options['lstm'].get(
            'use_skip_connections')
        if use_skip_connections:
            print("USING SKIP CONNECTIONS")
        lstm_outputs = []
        for lstm_num, lstm_input in enumerate(lstm_inputs):
            lstm_cells = []
            for i in range(n_lstm_layers):
                if projection_dim < lstm_dim:
                    # are projecting down output
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        lstm_dim, num_proj=projection_dim,
                        cell_clip=cell_clip, proj_clip=proj_clip)
                else:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        lstm_dim,
                        cell_clip=cell_clip, proj_clip=proj_clip)
                if use_skip_connections:
                    # ResidualWrapper adds inputs to outputs
                    if i == 0:
                        # don't add skip connection from token embedding to
                        # 1st layer output
                        pass
                    else:
                        # add a skip connection
                        lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)
                # add dropout
                if self.is_training:
                    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                              input_keep_prob=keep_prob)
                lstm_cells.append(lstm_cell)
            if n_lstm_layers > 1:
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
            else:
                lstm_cell = lstm_cells[0]
            with tf.control_dependencies([lstm_input]):
                self.init_lstm_state.append(
                    lstm_cell.zero_state(batch_size, DTYPE))
                # NOTE: this variable scope is for backward compatibility
                # with existing models...
                if self.bidirectional:
                    with tf.variable_scope('RNN_%s' % lstm_num):
                        _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                            lstm_cell,
                            tf.unstack(lstm_input, axis=1),
                            initial_state=self.init_lstm_state[-1])
                else:
                    _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                        lstm_cell,
                        tf.unstack(lstm_input, axis=1),
                        initial_state=self.init_lstm_state[-1])
                self.final_lstm_state.append(final_state)
            # (batch_size * unroll_steps, 512)
            lstm_output_flat = tf.reshape(
                tf.stack(_lstm_output_unpacked, axis=1), [-1, projection_dim])
            if self.is_training:
                # add dropout to output
                lstm_output_flat = tf.nn.dropout(lstm_output_flat,
                                                 keep_prob)
            tf.add_to_collection('lstm_output_embeddings',
                                 _lstm_output_unpacked)
            lstm_outputs.append(lstm_output_flat)
        return lstm_outputs

    def _transformer_lm(self, lstm_inputs):
        print('use transformer')
        lstm_outputs = []
        unroll_steps = tf.shape(lstm_inputs[0])[1]
        dropout = self.options['dropout']
        keep_prob = 1.0 - dropout
        params = tf.contrib.training.HParams(
            num_decoder_layers=self.options["transformer"].get("num_decoder_layers"),
            layer_preprocess=self.options["transformer"].get("layer_preprocess"),
            num_heads=self.options["transformer"].get("num_heads"),
            hidden_size=self.options["transformer"].get("hidden_size"),
            attention_dropout=self.options["transformer"].get("attention_dropout"),
            residual_dropout=self.options["transformer"].get("residual_dropout"),
            filter_size=self.options["transformer"].get("filter_size"),
            relu_dropout=self.options["transformer"].get("relu_dropout"),
            attention_key_channels=self.options['transformer'].get('hidden_size'),
            attention_value_channels=self.options['transformer'].get('hidden_size')
        )
        if not self.is_training:
            params.attention_dropout = 0
            params.residual_dropout = 0
            params.relu_dropout = 0
        max_relative_dist = self.options['transformer'].get('max_relative_dist')
        print('max_relative_dist %s' % max_relative_dist)
        for lstm_num, lstm_input in enumerate(lstm_inputs):
            att_bias = attention_bias(unroll_steps, "causal")
            lstm_input = add_timing_signal(lstm_input)
            with tf.control_dependencies([lstm_input]):
                with tf.variable_scope("transformer_%s" % lstm_num):
                    lstm_output = transformer_decoder(lstm_input, None, att_bias, None, params, max_relative_dist=max_relative_dist)
            num_context_steps = self.options.get('num_context_steps', 0)
            print('num_context_steps: %s' % num_context_steps)
            if num_context_steps > 0:
                lstm_output = lstm_output[:, num_context_steps:, :]
            tf.add_to_collection('lstm_output_embeddings', lstm_output)
            lstm_output_flat = tf.reshape(lstm_output, [-1, params.hidden_size])
            if self.is_training and not self.options['transformer'].get('no_additional_dropout'):
                # add dropout to output
                lstm_output_flat = tf.nn.dropout(lstm_output_flat,
                                                 keep_prob)

            lstm_outputs.append(lstm_output_flat)
        return lstm_outputs

    def _build_loss(self, lstm_outputs):
        '''
        Create:
            self.total_loss: total loss op for training
            self.softmax_W, softmax_b: the softmax variables
            self.next_token_id / _reverse: placeholders for gold input

        '''
        batch_size = self.options['batch_size']
        #unroll_steps = self.options['unroll_steps']

        n_tokens_vocab = self.options['n_tokens_vocab']

        # DEFINE next_token_id and *_reverse placeholders for the gold input
        def _get_next_token_placeholders(suffix):
            name = 'next_token_id' + suffix
            id_placeholder = tf.placeholder(DTYPE_INT,
                                   shape=(batch_size, None),
                                   name=name)
            return id_placeholder
        
        def _get_seq_length_placeholders(suffix):
            name = 'seq_length' + suffix
            return tf.placeholder(DTYPE_INT,
                                shape=(batch_size),
                                name=name)

        # get the window and weight placeholders
        self.next_token_id = _get_next_token_placeholders('')
        self.seq_length = _get_seq_length_placeholders('')
        if self.bidirectional:
            self.next_token_id_reverse = _get_next_token_placeholders(
                '_reverse')
            self.seq_length_reverse = _get_seq_length_placeholders('_reverse')
        
        
        # DEFINE THE SOFTMAX VARIABLES
        # get the dimension of the softmax weights
        # softmax dimension is the size of the output projection_dim
        softmax_dim = self.options['lstm']['projection_dim']

        # the output softmax variables -- they are shared if bidirectional
        if self.share_embedding_softmax:
            # softmax_W is just the embedding layer
            self.softmax_W = self.embedding_weights

        if self.options.get('store_embedding_in_gpu', False):
            with tf.variable_scope('softmax'):
                softmax_init = tf.random_normal_initializer(0.0,
                    1.0 / np.sqrt(softmax_dim))
                if not self.share_embedding_softmax:
                    self.softmax_W = tf.get_variable(
                        'W', [n_tokens_vocab, softmax_dim],
                        dtype=DTYPE,
                        initializer=softmax_init
                    )
                self.softmax_b = tf.get_variable(
                    'b', [n_tokens_vocab],
                    dtype=DTYPE,
                    initializer=tf.constant_initializer(0.0))
        else:
            with tf.variable_scope('softmax'), tf.device('/cpu:0'):
                # Glorit init (std=(1.0 / sqrt(fan_in))
                softmax_init = tf.random_normal_initializer(0.0,
                    1.0 / np.sqrt(softmax_dim))
                if not self.share_embedding_softmax:
                    self.softmax_W = tf.get_variable(
                        'W', [n_tokens_vocab, softmax_dim],
                        dtype=DTYPE,
                        initializer=softmax_init
                    )
                self.softmax_b = tf.get_variable(
                    'b', [n_tokens_vocab],
                    dtype=DTYPE,
                    initializer=tf.constant_initializer(0.0))

        # now calculate losses
        # loss for each direction of the LSTM
        self.individual_losses = []
        seq_lengths = []
        if self.bidirectional:
            next_ids = [self.next_token_id, self.next_token_id_reverse]
            seq_lengths = [self.seq_length, self.seq_length_reverse]
        else:
            next_ids = [self.next_token_id]
            seq_lengths = [self.seq_length]
        
        def _get_mask(seq_length, max_length):
            batch_size = tf.shape(seq_length)[0]
            mat = tf.reshape(tf.tile(tf.range(max_length), [batch_size]), [batch_size, max_length])
            length = tf.reshape(tf.cast(seq_length, dtype=tf.int32), [-1, 1])
            mask = tf.cast(mat < length, dtype=DTYPE)
            return mask

        for id_placeholder, lstm_output_flat, seq_length in zip(next_ids, lstm_outputs, seq_lengths):
            # flatten the LSTM output and next token id gold to shape:
            # (batch_size * unroll_steps, softmax_dim)
            # Flatten and reshape the token_id placeholders
            next_token_id_flat = tf.reshape(id_placeholder, [-1, 1])
            mask = _get_mask(seq_length, tf.shape(id_placeholder)[1])
            mask_flat = tf.reshape(mask, [-1, 1])

            with tf.control_dependencies([lstm_output_flat]):
                if self.is_training and self.sample_softmax:
                    losses = tf.nn.sampled_softmax_loss(
                                   self.softmax_W, self.softmax_b,
                                   next_token_id_flat, lstm_output_flat,
                                   self.options['n_negative_samples_batch'],
                                   self.options['n_tokens_vocab'],
                                   num_true=1)

                else:
                    # get the full softmax loss
                    output_scores = tf.matmul(
                        lstm_output_flat,
                        tf.transpose(self.softmax_W)
                    ) + self.softmax_b
                    # NOTE: tf.nn.sparse_softmax_cross_entropy_with_logits
                    #   expects unnormalized output since it performs the
                    #   softmax internally
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=output_scores,
                        labels=tf.squeeze(next_token_id_flat, squeeze_dims=[1])
                    )
                
                losses = losses * mask_flat

            self.individual_losses.append(tf.reduce_mean(losses))

        # now make the total loss -- it's the mean of the individual losses
        if self.bidirectional:
            self.total_loss = 0.5 * (self.individual_losses[0]
                                    + self.individual_losses[1])
        else:
            self.total_loss = self.individual_losses[0]


def average_gradients(tower_grads, batch_size, options):
    # calculate average gradient for each shared variable across all GPUs
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        # We need to average the gradients across each GPU.

        g0, v0 = grad_and_vars[0]

        if g0 is None:
            # no gradient for this variable, skip it
            average_grads.append((g0, v0))
            continue

        if isinstance(g0, tf.IndexedSlices):
            # If the gradient is type IndexedSlices then this is a sparse
            #   gradient with attributes indices and values.
            # To average, need to concat them individually then create
            #   a new IndexedSlices object.
            indices = []
            values = []
            for g, v in grad_and_vars:
                indices.append(g.indices)
                values.append(g.values)
            all_indices = tf.concat(indices, 0)
            avg_values = tf.concat(values, 0) / len(grad_and_vars)
            # deduplicate across indices
            av, ai = _deduplicate_indexed_slices(avg_values, all_indices)
            grad = tf.IndexedSlices(av, ai, dense_shape=g0.dense_shape)

        else:
            # a normal tensor can just do a simple average
            grads = []
            for g, v in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over 
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

        # the Variables are redundant because they are shared
        # across towers. So.. just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)

    assert len(average_grads) == len(list(zip(*tower_grads)))
    
    return average_grads


def summary_gradient_updates(grads, opt, lr):
    '''get summary ops for the magnitude of gradient updates'''

    # strategy:
    # make a dict of variable name -> [variable, grad, adagrad slot]
    vars_grads = {}
    for v in tf.trainable_variables():
        vars_grads[v.name] = [v, None, None]
    for g, v in grads:
        vars_grads[v.name][1] = g
        vars_grads[v.name][2] = opt.get_slot(v, 'accumulator')

    # now make summaries
    ret = []
    for vname, (v, g, a) in vars_grads.items():

        if g is None:
            continue

        if isinstance(g, tf.IndexedSlices):
            # a sparse gradient - only take norm of params that are updated
            values = tf.gather(v, g.indices)
            updates = lr * g.values
            if a is not None:
                updates /= tf.sqrt(tf.gather(a, g.indices))
        else:
            values = v
            updates = lr * g
            if a is not None:
                updates /= tf.sqrt(a)

        values_norm = tf.sqrt(tf.reduce_sum(v * v)) + 1.0e-7
        updates_norm = tf.sqrt(tf.reduce_sum(updates * updates))
        ret.append(
            tf.summary.scalar('UPDATE/' + vname.replace(':', '_'), updates_norm / values_norm))

    return ret

def _deduplicate_indexed_slices(values, indices):
    """Sums `values` associated with any non-unique `indices`.
    Args:
      values: A `Tensor` with rank >= 1.
      indices: A one-dimensional integer `Tensor`, indexing into the first
      dimension of `values` (as in an IndexedSlices object).
    Returns:
      A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
      de-duplicated version of `indices` and `summed_values` contains the sum of
      `values` slices associated with each unique index.
    """
    unique_indices, new_index_positions = tf.unique(indices)
    summed_values = tf.unsorted_segment_sum(
      values, new_index_positions,
      tf.shape(unique_indices)[0])
    return (summed_values, unique_indices)


def _get_feed_dict_from_X(X, start, end, model, char_inputs, bidirectional):
    feed_dict = {}
    if not char_inputs:
        token_ids = X['token_ids'][start:end]
        feed_dict[model.token_ids] = token_ids
    else:
        # character inputs
        char_ids = X['tokens_characters'][start:end]
        feed_dict[model.tokens_characters] = char_ids

    if bidirectional:
        if not char_inputs:
            feed_dict[model.token_ids_reverse] = \
                X['token_ids_reverse'][start:end]
        else:
            feed_dict[model.tokens_characters_reverse] = \
                X['tokens_characters_reverse'][start:end]

    # now the targets with weights
    next_id_placeholders = [[model.next_token_id, '']]
    if bidirectional:
        next_id_placeholders.append([model.next_token_id_reverse, '_reverse'])

    for id_placeholder, suffix in next_id_placeholders:
        name = 'next_token_id' + suffix
        feed_dict[id_placeholder] = X[name][start:end]
    
    feed_dict[model.seq_length] = X['seq_length'][start:end]
    
    if bidirectional:
        feed_dict[model.seq_length_reverse] = X['seq_length_reverse'][start:end]
    
    return feed_dict


def train(options, data, n_gpus, tf_save_dir, tf_log_dir,
          restart_ckpt_file=None, validation=None):

    # not restarting so save the options
    if restart_ckpt_file is None:
        with open(os.path.join(tf_save_dir, 'options.json'), 'w') as fout:
            fout.write(json.dumps(options))

    batch_size = options['batch_size']
    unroll_steps = options['unroll_steps']
    n_train_tokens = options.get('n_train_tokens', 768648884)
    n_tokens_per_batch = batch_size * unroll_steps * n_gpus
    n_batches_per_epoch = int(n_train_tokens / n_tokens_per_batch)
    n_batches_total = options['n_epochs'] * n_batches_per_epoch
    print("Training for %s epochs and %s batches" % (
        options['n_epochs'], n_batches_total))

    with tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        #global_step = tf.train.get_or_create_global_step()

        # set up the optimizer
        #lr = options.get('learning_rate', 0.2)
        opt_name = options.get('optimizer', 'adagrad')
        print('Use %s optimizer' % opt_name)
        opt_options = options[opt_name]
        lr = get_learning_rate(
            opt_options,
            global_step,
            n_batches_total,
            options['lstm']['projection_dim']
        )
        if opt_name == 'adagrad':
            opt = tf.train.AdagradOptimizer(
                learning_rate=lr,
                initial_accumulator_value=1.0)
        elif opt_name == 'adam':
            opt = tf.train.AdamOptimizer(
                lr,
                beta1=opt_options['beta1'],
                beta2=opt_options['beta2'],
                epsilon=opt_options['epsilon']
            )
        elif opt_name == 'lazyadam':
            opt = tf.contrib.opt.LazyAdamOptimizer(
                lr,
                beta1=opt_options['beta1'],
                beta2=opt_options['beta2'],
                epsilon=opt_options['epsilon']
            )


        # calculate the gradients on each GPU
        tower_grads = []
        models = []
        train_perplexity = tf.get_variable(
            'train_perplexity', [],
            initializer=tf.constant_initializer(0.0), trainable=False)
        norm_summaries = []
        for k in range(n_gpus):
            with tf.device('/gpu:%d' % k):
                with tf.variable_scope('lm', reuse=k > 0):
                    # calculate the loss for one model replica and get
                    #   lstm states
                    model = LanguageModel(options, True)
                    loss = model.total_loss
                    models.append(model)
                    # get gradients
                    grads = opt.compute_gradients(
                        loss * options['unroll_steps'],
                        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                    )
                    tower_grads.append(grads)
                    # keep track of loss across all GPUs
                    train_perplexity += loss

        #print_variable_summary()

        # calculate the mean of each gradient across all GPUs
        grads = average_gradients(tower_grads, options['batch_size'], options)
        grads, norm_summary_ops = clip_grads(grads, options, True, global_step)
        norm_summaries.extend(norm_summary_ops)

        # log the training perplexity
        train_perplexity = tf.exp(train_perplexity / n_gpus)
        perplexity_summmary = tf.summary.scalar(
            'train_perplexity', train_perplexity)

        # some histogram summaries.  all models use the same parameters
        # so only need to summarize one
        histogram_summaries = [
            tf.summary.histogram('token_embedding', models[0].embedding)
        ]
        # tensors of the output from the LSTM layer
        lstm_out = tf.get_collection('lstm_output_embeddings')
        histogram_summaries.append(
                tf.summary.histogram('lstm_embedding_0', lstm_out[0]))
        if options.get('bidirectional', False):
            # also have the backward embedding
            histogram_summaries.append(
                tf.summary.histogram('lstm_embedding_1', lstm_out[1]))

        # apply the gradients to create the training operation
        train_op = opt.apply_gradients(grads, global_step=global_step)

        # histograms of variables
        for v in tf.global_variables():
            histogram_summaries.append(tf.summary.histogram(v.name.replace(':', '_'), v))

        # get the gradient updates -- these aren't histograms, but we'll
        # only update them when histograms are computed
        histogram_summaries.extend(
            summary_gradient_updates(grads, opt, lr))

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        summary_op = tf.summary.merge(
            [perplexity_summmary] + norm_summaries
        )
        hist_summary_op = tf.summary.merge(histogram_summaries)

        init = tf.initialize_all_variables()

    # do the training loop
    bidirectional = options.get('bidirectional', False)
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True)) as sess:
        sess.run(init)

        # load the checkpoint data if needed
        if restart_ckpt_file is not None:
            loader = tf.train.Saver()
            loader.restore(sess, restart_ckpt_file)

        summary_writer = tf.summary.FileWriter(tf_log_dir, sess.graph)

        prev_steps = np.int64(sess.run(global_step))

        print ('Previous trained steps: %s' % prev_steps)

        # For each batch:
        # Get a batch of data from the generator. The generator will
        # yield batches of size batch_size * n_gpus that are sliced
        # and fed for each required placeholer.
        #
        # We also need to be careful with the LSTM states.  We will
        # collect the final LSTM states after each batch, then feed
        # them back in as the initial state for the next batch

        batch_size = options['batch_size']
        unroll_steps = options['unroll_steps']
        n_train_tokens = options.get('n_train_tokens', 768648884)
        n_tokens_per_batch = batch_size * unroll_steps * n_gpus
        n_batches_per_epoch = int(n_train_tokens / n_tokens_per_batch)
        n_batches_total = options['n_epochs'] * n_batches_per_epoch
        print("Training for %s epochs and %s batches" % (
            options['n_epochs'], n_batches_total))
        use_transformer = options.get('use_transformer', False)
        feed_dict = {}
        char_inputs = 'char_cnn' in options

        if not use_transformer:
            # get the initial lstm states
            init_state_tensors = []
            final_state_tensors = []
            for model in models:
                init_state_tensors.extend(model.init_lstm_state)
                final_state_tensors.extend(model.final_lstm_state)

            if char_inputs:
                max_chars = options['char_cnn']['max_characters_per_token']

            if not char_inputs:
                feed_dict = {
                    model.token_ids:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
                    for model in models
                }
            else:
                feed_dict = {
                    model.tokens_characters:
                        np.zeros([batch_size, unroll_steps, max_chars],
                                dtype=np.int32)
                    for model in models
                }

            if bidirectional:
                if not char_inputs:
                    feed_dict.update({
                        model.token_ids_reverse:
                            np.zeros([batch_size, unroll_steps], dtype=np.int64)
                        for model in models
                    })
                else:
                    feed_dict.update({
                        model.tokens_characters_reverse:
                            np.zeros([batch_size, unroll_steps, max_chars],
                                    dtype=np.int32)
                        for model in models
                    })

            init_state_values = sess.run(init_state_tensors, feed_dict=feed_dict)

        t1 = time.time()
        num_context_steps = options['num_context_steps'] if use_transformer else 0
        data_gen = data.iter_batches(batch_size * n_gpus, unroll_steps, num_context_steps=num_context_steps)
        best_valid_ppl = -1
        for batch_no, batch in enumerate(data_gen, start=1 + prev_steps):

            # slice the input in the batch for the feed_dict
            X = batch
            if not use_transformer:
                feed_dict = {t: v for t, v in zip(
                                            init_state_tensors, init_state_values)}
            for k in range(n_gpus):
                model = models[k]
                start = k * batch_size
                end = (k + 1) * batch_size

                feed_dict.update(
                    _get_feed_dict_from_X(X, start, end, model,
                                          char_inputs, bidirectional)
                )

            # This runs the train_op, summaries and the "final_state_tensors"
            #   which just returns the tensors, passing in the initial
            #   state tensors, token ids and next token ids
            if use_transformer:
                final_state_tensors = []
            if batch_no % 1250 != 0:
                
                ret = sess.run(
                    [train_op, summary_op, train_perplexity] +
                                                final_state_tensors,
                    feed_dict=feed_dict
                )

                # first three entries of ret are:
                #  train_op, summary_op, train_perplexity
                # last entries are the final states -- set them to
                # init_state_values
                # for next batch
                init_state_values = ret[3:] if not use_transformer else []

            else:
                # also run the histogram summaries
                ret = sess.run(
                    [train_op, summary_op, train_perplexity, hist_summary_op] + 
                                                final_state_tensors,
                    feed_dict=feed_dict
                )
                init_state_values = ret[4:] if not use_transformer else []

            if (batch_no % 200 == 0 or batch_no == n_batches_total) and validation is not None:
                validation.reset()
                valid_loss = _run_test(models[0], validation, options, sess, batch_size)
                valid_ppl = np.exp(valid_loss)
                if best_valid_ppl < 0 or valid_ppl < best_valid_ppl:
                    best_valid_ppl = valid_ppl
                    saver.save(sess, os.path.join(tf_save_dir, 'model.ckpt.bestvalid'))
                print('Best validation PPL: %s' % best_valid_ppl)
            if batch_no % 1250 == 0:
                summary_writer.add_summary(ret[3], batch_no)
            if batch_no % 100 == 0:
                # write the summaries to tensorboard and display perplexity
                summary_writer.add_summary(ret[1], batch_no)
                print("Batch %s, train_perplexity=%s" % (batch_no, ret[2]))
                print("Total time: %s" % (time.time() - t1))

            if (batch_no % 1250 == 0) or (batch_no == n_batches_total):
                # save the model
                checkpoint_path = os.path.join(tf_save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)

            if batch_no == n_batches_total:
                # done training!
                break


def clip_by_global_norm_summary(t_list, clip_norm, norm_name, variables):
    # wrapper around tf.clip_by_global_norm that also does summary ops of norms

    # compute norms
    # use global_norm with one element to handle IndexedSlices vs dense
    norms = [tf.global_norm([t]) for t in t_list]

    # summary ops before clipping
    summary_ops = []
    for ns, v in zip(norms, variables):
        name = 'norm_pre_clip/' + v.name.replace(':', '_')
        summary_ops.append(tf.summary.scalar(name, ns))

    # clip 
    clipped_t_list, tf_norm = tf.clip_by_global_norm(t_list, clip_norm)

    # summary ops after clipping
    norms_post = [tf.global_norm([t]) for t in clipped_t_list]
    for ns, v in zip(norms_post, variables):
        name = 'norm_post_clip/' + v.name.replace(':','_')
        summary_ops.append(tf.summary.scalar(name, ns))

    summary_ops.append(tf.summary.scalar(norm_name, tf_norm))

    return clipped_t_list, tf_norm, summary_ops


def clip_grads(grads, options, do_summaries, global_step):
    # grads = [(grad1, var1), (grad2, var2), ...]
    def _clip_norms(grad_and_vars, val, name):
        # grad_and_vars is a list of (g, v) pairs
        grad_tensors = [g for g, v in grad_and_vars]
        vv = [v for g, v in grad_and_vars]
        scaled_val = val
        if do_summaries:
            clipped_tensors, g_norm, so = clip_by_global_norm_summary(
                grad_tensors, scaled_val, name, vv)
        else:
            so = []
            clipped_tensors, g_norm = tf.clip_by_global_norm(
                grad_tensors, scaled_val)

        ret = []
        for t, (g, v) in zip(clipped_tensors, grad_and_vars):
            ret.append((t, v))

        return ret, so

    all_clip_norm_val = options['all_clip_norm_val']
    ret, summary_ops = _clip_norms(grads, all_clip_norm_val, 'norm_grad')

    assert len(ret) == len(grads)

    return ret, summary_ops


def test(options, ckpt_file, data, batch_size=256):
    '''
    Get the test set perplexity!
    '''
    if options.get('use_transformer', False):
        return test_transformer(options, ckpt_file, data, batch_size)

    bidirectional = options.get('bidirectional', False)
    char_inputs = 'char_cnn' in options
    if char_inputs:
        max_chars = options['char_cnn']['max_characters_per_token']

    unroll_steps = 1

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'), tf.variable_scope('lm'):
            test_options = dict(options)
            # NOTE: the number of tokens we skip in the last incomplete
            # batch is bounded above batch_size * unroll_steps
            test_options['batch_size'] = batch_size
            test_options['unroll_steps'] = None
            model = LanguageModel(test_options, False)
            # we use the "Saver" class to load the variables
            loader = tf.train.Saver()
            loader.restore(sess, ckpt_file)

        # model.total_loss is the op to compute the loss
        # perplexity is exp(loss)
        init_state_tensors = model.init_lstm_state
        final_state_tensors = model.final_lstm_state
        if not char_inputs:
            feed_dict = {
                model.token_ids:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
            }
            if bidirectional:
                feed_dict.update({
                    model.token_ids_reverse:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
                })  
        else:
            feed_dict = {
                model.tokens_characters:
                   np.zeros([batch_size, unroll_steps, max_chars],
                                 dtype=np.int32)
            }
            if bidirectional:
                feed_dict.update({
                    model.tokens_characters_reverse:
                        np.zeros([batch_size, unroll_steps, max_chars],
                            dtype=np.int32)
                })

        init_state_values = sess.run(
            init_state_tensors,
            feed_dict=feed_dict)

        t1 = time.time()
        batch_losses = []
        total_loss = 0.0
        for batch_no, batch in enumerate(
                                data.iter_batches(batch_size, 1), start=1):
            # slice the input in the batch for the feed_dict
            X = batch

            feed_dict = {t: v for t, v in zip(
                                        init_state_tensors, init_state_values)}

            feed_dict.update(
                _get_feed_dict_from_X(X, 0, X['token_ids'].shape[0], model, 
                                          char_inputs, bidirectional)
            )

            ret = sess.run(
                [model.total_loss, final_state_tensors],
                feed_dict=feed_dict
            )

            loss, init_state_values = ret
            batch_losses.append(loss)
            batch_perplexity = np.exp(loss)
            total_loss += loss
            avg_perplexity = np.exp(total_loss / batch_no)

            print("batch=%s, batch_perplexity=%s, avg_perplexity=%s, time=%s" %
                (batch_no, batch_perplexity, avg_perplexity, time.time() - t1))
            
    avg_loss = np.mean(batch_losses)
    print("FINSIHED!  AVERAGE PERPLEXITY = %s" % np.exp(avg_loss))

    with open(ckpt_file + '.eval.txt', 'w') as fout:
        fout.write("EVAL PERPLEXITY = %s" % np.exp(avg_loss))

    return np.exp(avg_loss)

def test_transformer(options, ckpt_file, data, batch_size):
    '''
    Get the test set perplexity!
    '''
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'), tf.variable_scope('lm'):
            test_options = dict(options)
            # NOTE: the number of tokens we skip in the last incomplete
            # batch is bounded above batch_size * unroll_steps
            test_options['batch_size'] = batch_size
            test_options['unroll_steps'] = None
            model = LanguageModel(test_options, False)
            # we use the "Saver" class to load the variables
            loader = tf.train.Saver()
            loader.restore(sess, ckpt_file)

        avg_loss = _run_test(model, data, options, sess, batch_size)

    with open(ckpt_file + '.eval.txt', 'w') as fout:
        fout.write("EVAL PERPLEXITY = %s" % np.exp(avg_loss))
    return np.exp(avg_loss)


def _run_test(model, data, options, sess, batch_size):
    feed_dict = {}
    t1 = time.time()
    total_loss = 0.0
    total_num_words = 0
    total_loss_reverse = 0.0
    total_num_words_reverse = 0
    char_inputs = 'char_cnn' in options
    bidirectional = options.get('bidirectional', False)
    for batch_no, batch in enumerate(
                            data.iter_batches(batch_size, options['unroll_steps'], num_context_steps=options.get('num_context_steps', 0))):
        # slice the input in the batch for the feed_dict
        X = batch
        feed_dict = _get_feed_dict_from_X(X, 0, batch_size, model, char_inputs, bidirectional)
        ret = sess.run(
            model.individual_losses,
            feed_dict=feed_dict
        )
        loss = ret[0]
        num_words = sum(X['seq_length'])
        total_loss += loss * num_words
        total_num_words += num_words
        num_words_reverse = sum(X['seq_length_reverse']) if bidirectional else 0
        loss_reverse = ret[1] if bidirectional else 0
        batch_perplexity = np.exp((loss * num_words + loss_reverse * num_words_reverse) / (num_words + num_words_reverse))
        total_num_words_reverse += num_words_reverse
        total_loss_reverse += loss_reverse * num_words_reverse
        avg_perplexity = np.exp((total_loss + total_loss_reverse) / (total_num_words + total_num_words_reverse))

        print("batch=%s, batch_perplexity=%s, avg_perplexity=%s, time=%s" %
            (batch_no, batch_perplexity, avg_perplexity, time.time() - t1))
    
    avg_loss = (total_loss + total_loss_reverse) / (total_num_words + total_num_words_reverse)
    print("Total words %s" % (total_num_words + total_num_words_reverse))
    print("FINSIHED!  AVERAGE PERPLEXITY = %s" % np.exp(avg_loss))
    if bidirectional:
        ppl_forward = np.exp(total_loss / total_num_words)
        ppl_backward = np.exp(total_loss_reverse/ total_num_words_reverse)
        print('AVERAGE FORWARD PERPLEXITY = %s' % ppl_forward)
        print('AVERAGE BACKWARD PERPLEXITY = %s' % ppl_backward)
    return avg_loss


def load_options_latest_checkpoint(tf_save_dir):
    options_file = os.path.join(tf_save_dir, 'options.json')
    ckpt_file = tf.train.latest_checkpoint(tf_save_dir)

    with open(options_file, 'r') as fin:
        options = json.load(fin)

    return options, ckpt_file


def load_vocab(vocab_file, max_word_length=None):
    if max_word_length:
        return UnicodeCharsVocabulary(vocab_file, max_word_length,
                                      validate_file=True)
    else:
        return Vocabulary(vocab_file, validate_file=True)


def dump_weights(tf_save_dir, outfile):
    '''
    Dump the trained weights from a model to a HDF5 file.
    '''
    import h5py

    def _get_outname(tf_name):
        outname = re.sub(':0$', '', tf_name)
        outname = outname.lstrip('lm/')
        outname = re.sub('/rnn/', '/RNN/', outname)
        outname = re.sub('/multi_rnn_cell/', '/MultiRNNCell/', outname)
        outname = re.sub('/cell_', '/Cell', outname)
        outname = re.sub('/lstm_cell/', '/LSTMCell/', outname)
        if '/RNN/' in outname:
            if 'projection' in outname:
                outname = re.sub('projection/kernel', 'W_P_0', outname)
            else:
                outname = re.sub('/kernel', '/W_0', outname)
                outname = re.sub('/bias', '/B', outname)
        return outname

    options, ckpt_file = load_options_latest_checkpoint(tf_save_dir)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.variable_scope('lm'):
            model = LanguageModel(options, False)
            # we use the "Saver" class to load the variables
            loader = tf.train.Saver()
            loader.restore(sess, ckpt_file)

        with h5py.File(outfile, 'w') as fout:
            for v in tf.trainable_variables():
                if v.name.find('softmax') >= 0:
                    # don't dump these
                    continue
                outname = _get_outname(v.name)
                print("Saving variable {0} with name {1}".format(
                    v.name, outname))
                shape = v.get_shape().as_list()
                dset = fout.create_dataset(outname, shape, dtype='float32')
                values = sess.run([v])[0]
                dset[...] = values

