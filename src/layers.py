#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 05/2/2018 10:10 PM

import math
import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.contrib.layers import fully_connected as fc
from tensorflow.contrib.layers import layer_norm
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import xavier_initializer as xa_init, variance_scaling_initializer as he_init

bias_init = lambda: tf.constant_initializer(value=0, dtype=tf.float32)


def mask_logits(inputs, mask, mask_value=-1e5):
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)


def conv(inputs, hidden_size, kernel_size, bias=True, activation=tf.nn.relu, padding='VALID', scope='conv', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        in_channels = inputs.get_shape()[-1]
        filter_ = tf.get_variable('filter', shape=[1, kernel_size, in_channels, hidden_size],
                                  initializer=he_init(), dtype=tf.float32)
        strides = [1, 1, 1, 1]
        out = tf.nn.conv2d(inputs, filter_, strides, padding)  # [N*M, JX, W/filter_stride, d]
        if bias:
            b = tf.get_variable('bias', shape=[hidden_size],
                                initializer=bias_init(), dtype=tf.float32)
            out += b
        if activation is not None:
            out = activation(out)
        out = tf.reduce_max(out, 2)
        return out


def depthwise_conv(inputs, hidden_size, kernel_size=1, bias=True, activation=None, scope='depthwise_conv', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        shapes = inputs.shape.as_list()
        depthwise_filter = tf.get_variable('depthwise_filter', (kernel_size, 1, shapes[-1], 1),
                                           dtype=tf.float32, initializer=he_init())
        pointwise_filter = tf.get_variable('pointwise_filter', (1, 1, shapes[-1], hidden_size),
                                           dtype=tf.float32, initializer=he_init())
        outputs = tf.nn.separable_conv2d(inputs, depthwise_filter, pointwise_filter,
                                         strides=(1, 1, 1, 1), padding='SAME')
        if bias:
            b = tf.get_variable('bias', outputs.shape[-1], initializer=bias_init())
            outputs += b
        if activation is not None:
            return activation(outputs)
        else:
            return outputs


def highway(x, size=None, activation=None, num_layers=2, dropout=0.0, scope='highway', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        if size is None:
            size = x.shape.as_list()[-1]
        else:
            x = fc(x, size, weights_initializer=xa_init(), biases_initializer=bias_init(),
                   activation_fn=None, scope='input_projection', reuse=reuse)
        for i in range(num_layers):
            T = fc(x, size, weights_initializer=xa_init(), biases_initializer=bias_init(),
                   activation_fn=tf.sigmoid, scope='gate_%d' % i, reuse=reuse)
            H = fc(x, size, weights_initializer=he_init(), biases_initializer=bias_init(),
                   activation_fn=activation, scope='activation_%d' % i, reuse=reuse)
            H = tf.nn.dropout(H, 1.0 - dropout)
            x = H * T + x * (1.0 - T)
        return x


def trilinear_similarity(x1, x2, scope='trilinear', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        x1_shape = x1.shape.as_list()
        x2_shape = x2.shape.as_list()
        if len(x1_shape) != 3 or len(x2_shape) != 3:
            raise ValueError('`args` must be 3 dims (batch_size, len, dimension)')
        if x1_shape[2] != x2_shape[2]:
            raise ValueError('the last dimension of `args` must equal')
        w1 = tf.get_variable('kernel_x1', [x1_shape[2], 1], dtype=x1.dtype, initializer=xa_init())
        w2 = tf.get_variable('kernel_x2', [x2_shape[2], 1], dtype=x2.dtype, initializer=xa_init())
        w3 = tf.get_variable('kernel_mul', [1, 1, x1_shape[2]], dtype=x2.dtype, initializer=xa_init())
        bias = tf.get_variable('bias', [1], dtype=x1.dtype, initializer=bias_init())
        r1 = tf.einsum('aij,jk->aik', x1, w1)
        r2 = tf.einsum('aij,jk->aki', x2, w2)
        r3 = tf.einsum('aij,akj->aik', x1 * w3, x2)
        return r1 + r2 + r3 + bias


def attn_pooling(pooling_vectors, hidden_size, ref_vector=None, mask=None, scope=None):
    with tf.variable_scope(scope or 'attn_pooling'):
        u = fc(pooling_vectors, num_outputs=hidden_size, activation_fn=None, biases_initializer=None)
        if ref_vector is not None:
            u += fc(tf.expand_dims(ref_vector, 1), num_outputs=hidden_size, activation_fn=None)
        logits = fc(tf.tanh(u), num_outputs=1, activation_fn=None)
        if mask is not None:
            logits = mask_logits(logits, mask=tf.expand_dims(mask, -1))
        scores = tf.nn.softmax(logits, 1)
        pooled_vector = tf.reduce_sum(pooling_vectors * scores, axis=1)
    return pooled_vector


def encoder_block(inputs, num_conv_layers, kernel_size, hidden_size, num_heads, num_blocks=1, mask=None,
                  dropout=0.0, use_relative_pos=False, scope='encoder_block', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = inputs
        for block_repeat_idx in range(num_blocks):
            with tf.variable_scope('block_%d' % block_repeat_idx, reuse=reuse):
                total_sublayers = num_conv_layers + 2
                sublayer = 0
                # position encoding
                if not use_relative_pos:
                    outputs += get_timing_signal_1d(tf.shape(outputs)[1], tf.shape(outputs)[2])
                # convolutions
                outputs = tf.expand_dims(outputs, 2)
                for i in range(num_conv_layers):
                    residual = outputs
                    outputs = layer_norm(outputs, begin_norm_axis=-1, begin_params_axis=-1,
                                         scope='conv_layer_norm_%d' % i, reuse=reuse)  # TODO: change layer norm
                    if i % 2 == 0:
                        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
                    if isinstance(kernel_size, list):
                        kernel_num = len(kernel_size)
                        kernel_outputs = [
                            depthwise_conv(outputs, hidden_size, bias=True, activation=tf.nn.relu,
                                           kernel_size=k, scope='depthwise_conv_%d_kernel_%d' % (i, k), reuse=reuse)
                            for k in kernel_size
                        ]
                        kernel_weights = tf.nn.softmax(
                            tf.get_variable('kernel_weights_conv_%d' % i, [kernel_num], dtype=tf.float32,
                                            trainable=True, initializer=tf.constant_initializer(1.0 / kernel_num)),
                            axis=0)
                        outputs = 0
                        for j in range(kernel_num):
                            outputs += kernel_outputs[j] * kernel_weights[j]
                    else:
                        outputs = depthwise_conv(outputs, hidden_size, bias=True,
                                                 activation=tf.nn.relu,
                                                 # activation=tf.nn.relu if i < num_conv_layers - 1 else None,
                                                 kernel_size=kernel_size, scope='depthwise_conv_%d' % i, reuse=reuse)
                    sublayer += 1
                    outputs = layer_dropout(residual, residual + tf.nn.dropout(outputs, 1 - dropout),
                                            dropout * float(sublayer) / total_sublayers)
                outputs = tf.squeeze(outputs, 2)
                # self attention
                residual = outputs
                outputs = layer_norm(outputs, begin_norm_axis=-1, begin_params_axis=-1,
                                     scope='self_attention_layer_norm', reuse=reuse)
                outputs = tf.nn.dropout(outputs, 1.0 - dropout)
                outputs = self_attention(outputs, hidden_size, num_heads,
                                         use_relative_pos=use_relative_pos, mask=mask,
                                         scope='self_attention_layer', reuse=reuse)
                sublayer += 1
                outputs = layer_dropout(residual, residual + tf.nn.dropout(outputs, 1 - dropout),
                                        dropout * float(sublayer) / total_sublayers)
                # feed forward
                residual = outputs
                outputs = layer_norm(outputs, begin_norm_axis=-1, begin_params_axis=-1,
                                     scope='fc_layer_norm', reuse=reuse)
                outputs = tf.nn.dropout(outputs, 1.0 - dropout)
                outputs = fc(outputs, hidden_size, tf.nn.relu,
                             weights_initializer=he_init(), biases_initializer=bias_init(),
                             scope='fc_layer_1', reuse=reuse)
                # outputs = tf.nn.dropout(outputs, 1 - dropout)
                outputs = fc(outputs, hidden_size, None,
                             weights_initializer=xa_init(), biases_initializer=bias_init(),
                             scope='fc_layer_2', reuse=reuse)
                sublayer += 1
                outputs = layer_dropout(residual, residual + tf.nn.dropout(outputs, 1 - dropout),
                                        dropout * float(sublayer) / total_sublayers)
        return outputs


def self_attention(inputs, hidden_size, num_heads, use_relative_pos=False, max_relative_position=16,
                   mask=None, scope='self_attention', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        Q = split_heads(
            fc(inputs, hidden_size, activation_fn=None,
               weights_initializer=xa_init(), biases_initializer=None, scope='q_projection', reuse=reuse),
            num_heads)
        K = split_heads(
            fc(inputs, hidden_size, activation_fn=None,
               weights_initializer=xa_init(), biases_initializer=None, scope='k_projection', reuse=reuse),
            num_heads)
        V = split_heads(
            fc(inputs, hidden_size, activation_fn=None,
               weights_initializer=xa_init(), biases_initializer=None, scope='v_projection', reuse=reuse),
            num_heads)
        Q *= (float(hidden_size) // num_heads) ** -0.5

        length = tf.shape(V)[2]
        depth = V.shape[3]

        # calculate similarity matrix
        if use_relative_pos:
            relations_keys = get_relative_positions_embeddings(length, depth, max_relative_position,
                                                               "relative_positions_keys")
            sim_logits = _relative_attention_inner(Q, K, relations_keys, transpose=True)
        else:
            sim_logits = tf.matmul(Q, K, transpose_b=True)
        if mask is not None:
            logit_mask = tf.expand_dims(tf.expand_dims(tf.cast(mask, tf.float32), 1), 1)
            sim_logits = mask_logits(sim_logits, logit_mask)

        # compute the attention output
        attn_weights = tf.nn.softmax(sim_logits, name='attention_weights')
        if use_relative_pos:
            relations_values = get_relative_positions_embeddings(length, depth, max_relative_position,
                                                                 "relative_positions_values")
            multi_head_attns = _relative_attention_inner(attn_weights, V, relations_values, transpose=False)
        else:
            multi_head_attns = tf.matmul(attn_weights, V)
        outputs = merge_heads(multi_head_attns)
        outputs = fc(outputs,
                     hidden_size, activation_fn=None,
                     weights_initializer=xa_init(), biases_initializer=None, scope='merge_projection', reuse=reuse)
        return outputs


def custom_dynamic_rnn(cell, inputs, inputs_len, initial_state=None):
    """
    Implements a dynamic rnn that can store scores in the pointer network,
    the reason why we implements this is that the raw_rnn or dynamic_rnn function in Tensorflow
    seem to require the hidden unit and memory unit has the same dimension, and we cannot
    store the scores directly in the hidden unit.
    Args:
        cell: RNN cell
        inputs: the input sequence to rnn
        inputs_len: valid length
        initial_state: initial_state of the cell
    Returns:
        outputs and state
    """
    batch_size = tf.shape(inputs)[0]
    max_time = tf.shape(inputs)[1]

    inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
    inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1, 0, 2]))
    emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    t0 = tf.constant(0, dtype=tf.int32)
    if initial_state is not None:
        s0 = initial_state
    else:
        s0 = cell.zero_state(batch_size, dtype=tf.float32)
    f0 = tf.zeros([batch_size], dtype=tf.bool)

    def loop_fn(t, prev_s, emit_ta, finished):
        """
        the loop function of rnn
        """
        cur_x = inputs_ta.read(t)
        scores, cur_state = cell(cur_x, prev_s)

        # copy through
        scores = tf.where(finished, tf.zeros_like(scores), scores)

        if isinstance(cell, tc.rnn.LSTMCell):
            cur_c, cur_h = cur_state
            prev_c, prev_h = prev_s
            cur_state = tc.rnn.LSTMStateTuple(tf.where(finished, prev_c, cur_c),
                                              tf.where(finished, prev_h, cur_h))
        else:
            cur_state = tf.where(finished, prev_s, cur_state)

        emit_ta = emit_ta.write(t, scores)
        finished = tf.greater_equal(t + 1, inputs_len)
        return [t + 1, cur_state, emit_ta, finished]

    _, state, emit_ta, _ = tf.while_loop(
        cond=lambda _1, _2, _3, finished: tf.logical_not(tf.reduce_all(finished)),
        body=loop_fn,
        loop_vars=(t0, s0, emit_ta, f0),
        parallel_iterations=32,
        swap_memory=False)

    outputs = tf.transpose(emit_ta.stack(), [1, 0, 2])
    return outputs, state


def recurrent_self_attention(inputs, hidden_size, num_heads, use_relative_pos=True, max_relative_position=16,
                             mask=None, scope='self_attention', reuse=None):
    pass


def _relative_attention_inner(x, y, z=None, transpose=False):
    """Relative position-aware dot-product attention inner calculation.

    This batches matrix multiply calculations to avoid unnecessary broadcasting.

    Args:
    x: Tensor with shape [batch_size, heads, length, length or depth].
    y: Tensor with shape [batch_size, heads, length, depth].
    z: Tensor with shape [length, length, depth].
    transpose: Whether to transpose inner matrices of y and z. Should be true if
        last dimension of x is depth, not length.

    Returns:
    A Tensor with shape [batch_size, heads, length, length or depth].
    """
    batch_size = tf.shape(x)[0]
    heads = x.get_shape().as_list()[1]
    length = tf.shape(x)[2]

    # xy_matmul is [batch_size, heads, length, length or depth]
    xy_matmul = tf.matmul(x, y, transpose_b=transpose)
    if z is not None:
        # x_t is [length, batch_size, heads, length or depth]
        x_t = tf.transpose(x, [2, 0, 1, 3])
        # x_t_r is [length, batch_size * heads, length or depth]
        x_t_r = tf.reshape(x_t, [length, heads * batch_size, -1])
        # x_tz_matmul is [length, batch_size * heads, length or depth]
        x_tz_matmul = tf.matmul(x_t_r, z, transpose_b=transpose)
        # x_tz_matmul_r is [length, batch_size, heads, length or depth]
        x_tz_matmul_r = tf.reshape(x_tz_matmul, [length, batch_size, heads, -1])
        # x_tz_matmul_r_t is [batch_size, heads, length, length or depth]
        x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r, [1, 2, 0, 3])
        return xy_matmul + x_tz_matmul_r_t
    else:
        return xy_matmul


def split_heads(x, n):
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return tf.transpose(ret, [0, 2, 1, 3])


def merge_heads(x):
    x = tf.transpose(x, [0, 2, 1, 3])
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)
    return ret


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


def get_relative_positions_embeddings(length, depth, max_relative_position, name):
    """Generates tensor of size [length, length, depth]."""
    with tf.variable_scope(name):
        relative_positions_matrix = get_relative_positions_matrix(
            length, max_relative_position)
        vocab_size = max_relative_position * 2 + 1
        # Generates embedding for each relative position of dimension depth.
        embeddings_table = tf.get_variable("embeddings", [vocab_size, depth])
        embeddings = tf.gather(embeddings_table, relative_positions_matrix)
        return embeddings


def get_relative_positions_matrix(length, max_relative_position):
    """Generates matrix of relative positions between inputs."""
    range_vec = tf.range(length)
    range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
    distance_mat = range_mat - tf.transpose(range_mat)
    distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                            max_relative_position)
    # Shift values to be >= 0. Each integer still uniquely identifies a relative
    # position difference.
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat


def layer_dropout(dropped, no_dropped, dropout_rate):
    pred = tf.random_uniform([]) < dropout_rate
    return tf.cond(pred, lambda: dropped, lambda: no_dropped)


def rnn(rnn_type, inputs, length, hidden_size, layer_num=1,
        dropout_keep_prob=None, concat=True, scope='rnn', reuse=None):
    """
    Implements (Bi-)LSTM, (Bi-)GRU and (Bi-)RNN
    Args:
        rnn_type: the type of rnn
        inputs: padded inputs into rnn
        length: the valid length of the inputs
        hidden_size: the size of hidden units
        layer_num: multiple rnn layer are stacked if layer_num > 1
        dropout_keep_prob:
        concat: When the rnn is bidirectional, the forward outputs and backward outputs are
                concatenated if this is True, else we add them.
        scope: name scope
        reuse: reuse variables in this scope
    Returns:
        RNN outputs and final state
    """
    with tf.variable_scope(scope, reuse=reuse):
        if not rnn_type.startswith('bi'):
            cell = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, states = tf.nn.dynamic_rnn(cell, inputs, sequence_length=length, dtype=tf.float32)
            if rnn_type.endswith('lstm'):
                c = [state.c for state in states]
                h = [state.h for state in states]
                states = h
        else:
            cell_fw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            cell_bw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_bw, cell_fw, inputs, sequence_length=length, dtype=tf.float32
            )
            states_fw, states_bw = states
            if rnn_type.endswith('lstm'):
                c_fw = [state_fw.c for state_fw in states_fw]
                h_fw = [state_fw.h for state_fw in states_fw]
                c_bw = [state_bw.c for state_bw in states_bw]
                h_bw = [state_bw.h for state_bw in states_bw]
                states_fw, states_bw = h_fw, h_bw
            if concat:
                outputs = tf.concat(outputs, 2)
                states = tf.concat([states_fw, states_bw], 1)
            else:
                outputs = outputs[0] + outputs[1]
                states = states_fw + states_bw
        return outputs, states


def get_cell(rnn_type, hidden_size, layer_num=1, dropout_keep_prob=None):
    """
    Gets the RNN Cell
    Args:
        rnn_type: 'lstm', 'gru' or 'rnn'
        hidden_size: The size of hidden units
        layer_num: MultiRNNCell are used if layer_num > 1
        dropout_keep_prob: dropout in RNN
    Returns:
        An RNN Cell
    """
    cells = []
    for i in range(layer_num):
        if rnn_type.endswith('lstm'):
            # cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
            cell = tc.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=hidden_size)
        elif rnn_type.endswith('gru'):
            # cell = tc.rnn.GRUCell(num_units=hidden_size)
            cell = tc.cudnn_rnn.CudnnCompatibleGRUCell(num_units=hidden_size)
        elif rnn_type.endswith('rnn'):
            cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
        else:
            raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
        if dropout_keep_prob is not None:
            cell = tc.rnn.DropoutWrapper(cell,
                                         input_keep_prob=dropout_keep_prob,
                                         output_keep_prob=dropout_keep_prob)
        cells.append(cell)
    cells = tc.rnn.MultiRNNCell(cells, state_is_tuple=True)
    return cells