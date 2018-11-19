#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 04/24/2018 8:11 PM

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as fc
from base_model import BaseModel
from layers import conv, highway, encoder_block, trilinear_similarity, mask_logits


class QANet(BaseModel):
    def _build_graph(self):
        with tf.variable_scope('input_layer'):
            self._inputs()
        with tf.variable_scope('input_embedding_layer'):
            self._embed()
        with tf.variable_scope('embedding_encoder_layer'):
            self._embedding_encoder()
        with tf.variable_scope('context_query_attention_layer'):
            self._attention()
        with tf.variable_scope('model_encoder_layer'):
            self._model_encoder()
        with tf.variable_scope('output_layer'):
            self._output()
        with tf.variable_scope('loss'):
            self._compute_loss()
        _, self.grad_norm, self.train_op = self._get_train_op(self.loss)

    def _train_epoch(self, train_batches, eval_func=None, eval_every_n_steps=None, print_every_n_steps=None):
        total_loss, total_batch_num, accum_loss = 0, 0, 0
        for bitx, batch in enumerate(train_batches, 1):
            feed_dict = {self.c: batch['context_word_ids'],
                         self.q: batch['question_word_ids'],
                         self.cc: batch['context_char_ids'],
                         self.qc: batch['question_char_ids'],
                         self.start: batch['start_pos'],
                         self.end: batch['end_pos'],
                         self.dropout: self.dropout_value}

            _, loss, step = self.sess.run([self.train_op, self.loss, self.global_step], feed_dict)

            accum_loss += loss
            if step != 0 and print_every_n_steps and step % print_every_n_steps == 0:
                self.logger.info('Global step: {}~{}, bitx: {}~{}, ave loss: {}'.format(
                    step - print_every_n_steps + 1, step,
                    bitx - print_every_n_steps + 1, bitx, accum_loss / print_every_n_steps))
                accum_loss = 0
            if eval_func is not None and eval_every_n_steps > 0 and step % eval_every_n_steps == 0:
                eval_func(step_cnt=step)
            total_loss += loss
            total_batch_num += 1
        return total_loss / total_batch_num

    def _predict_batch(self, batch):
        feed_dict = {self.c: batch['context_word_ids'],
                     self.q: batch['question_word_ids'],
                     self.cc: batch['context_char_ids'],
                     self.qc: batch['question_char_ids'],
                     self.dropout: 0.0}

        pred_starts, pred_ends = self.sess.run([self.pred_start, self.pred_end], feed_dict)

        batch_preds = []
        for sample_idx in range(len(batch['raw_data'])):
            sample_info = batch['raw_data'][sample_idx]
            pred_start, pred_end = pred_starts[sample_idx], pred_ends[sample_idx]
            if 'context_word_offsets' in sample_info:
                word_offsets = sample_info['context_word_offsets']
                answer = sample_info['context'][word_offsets[pred_start][0]: word_offsets[pred_end][1] + 1]
            else:
                self.logger.warning('No offsets information!')
                context_words = sample_info['context_words']
                answer = ' '.join(context_words[pred_start: pred_end + 1])
            batch_preds.append({'pred_answer': answer})
            # self.logger.info(' '.join(context_words))
            # self.logger.info(pred_start)
            # self.logger.info(pred_end)
            # self.logger.info(start_probs[sample_idx].tolist())
            # self.logger.info(end_probs[sample_idx].tolist())
        return batch_preds

    def _inputs(self):
        self.c = tf.placeholder(tf.int32, [None, None], 'context')
        self.q = tf.placeholder(tf.int32, [None, None], 'question')
        self.cc = tf.placeholder(tf.int32, [None, None, None], 'context_char')
        self.qc = tf.placeholder(tf.int32, [None, None, None], 'question_char')
        self.start = tf.placeholder(tf.int32, [None], 'start_idx')
        self.end = tf.placeholder(tf.int32, [None], 'end_idx')
        self.dropout = tf.placeholder_with_default(0.0, (), name='dropout_rate')

        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        self.batch_size = tf.shape(self.c)[0]
        self.padded_c_len = tf.shape(self.c)[1]
        self.padded_q_len = tf.shape(self.q)[1]

    def _embed(self):
        with tf.device('/cpu:0'):
            word_pad_emb = tf.get_variable('word_pad_embedding', shape=(1, self.word_vocab.embed_dim),
                                           initializer=tf.zeros_initializer, trainable=False)
            word_unk_emb = tf.get_variable('word_unk_embedding', shape=(1, self.word_vocab.embed_dim),
                                           initializer=tf.zeros_initializer, trainable=True)
            word_emb_init = tf.constant_initializer(self.word_vocab.embeddings[2:]) \
                if self.word_vocab.embeddings is not None \
                else tf.random_normal_initializer()
            normal_word_embs = tf.get_variable('normal_word_embeddings',
                                               shape=(self.word_vocab.size() - 2, self.word_vocab.embed_dim),
                                               initializer=word_emb_init,
                                               trainable=False)
            self.word_emb_mat = tf.concat([word_pad_emb, word_unk_emb, normal_word_embs], 0)
            char_pad_emb = tf.get_variable('char_pad_embedding', shape=(1, self.char_vocab.embed_dim),
                                           initializer=tf.zeros_initializer, trainable=False)
            char_emb_init = tf.constant_initializer(self.char_vocab.embeddings[1:]) \
                if self.char_vocab.embeddings is not None \
                else tf.random_normal_initializer()
            normal_char_embs = tf.get_variable('normal_char_embeddings',
                                               shape=(self.char_vocab.size() - 1, self.char_vocab.embed_dim),
                                               initializer=char_emb_init,
                                               trainable=True)
            self.char_emb_mat = tf.concat([char_pad_emb, normal_char_embs], 0)
            self.emb_c = tf.nn.dropout(tf.nn.embedding_lookup(self.word_emb_mat, self.c), 1.0 - self.dropout)
            self.emb_q = tf.nn.dropout(tf.nn.embedding_lookup(self.word_emb_mat, self.q), 1.0 - self.dropout)

            self.emb_cc = tf.nn.dropout(tf.nn.embedding_lookup(self.char_emb_mat, self.cc), 1.0 - 0.5 * self.dropout)
            self.emb_qc = tf.nn.dropout(tf.nn.embedding_lookup(self.char_emb_mat, self.qc), 1.0 - 0.5 * self.dropout)

        # check the paper, it seems to use another operation
        # self.conv_emb_cc = conv(self.emb_cc, self.hidden_size, kernel_size=5, activation=tf.nn.relu, reuse=None)
        # self.conv_emb_qc = conv(self.emb_qc, self.hidden_size, kernel_size=5, activation=tf.nn.relu, reuse=True)
        self.conv_emb_cc = tf.reduce_max(self.emb_cc, 2)
        self.conv_emb_qc = tf.reduce_max(self.emb_qc, 2)
        self.conv_emb_cc = fc(self.conv_emb_cc, self.char_vocab.embed_dim, activation_fn=None)
        self.conv_emb_qc = fc(self.conv_emb_qc, self.char_vocab.embed_dim, activation_fn=None)

        self.emb_c = highway(tf.concat([self.emb_c, self.conv_emb_cc], axis=2),
                             size=self.hidden_size, dropout=self.dropout, num_layers=2, scope='highway', reuse=None)
        self.emb_q = highway(tf.concat([self.emb_q, self.conv_emb_qc], axis=2),
                             size=self.hidden_size, dropout=self.dropout, num_layers=2, scope='highway', reuse=True)

    def _embedding_encoder(self):
        # TODO: check the paper, it uses conv here
        self.emb_c = fc(self.emb_c, self.hidden_size, activation_fn=None, scope='input_projection', reuse=None)
        self.emb_q = fc(self.emb_q, self.hidden_size, activation_fn=None, scope='input_projection', reuse=True)

        # TODO: consider to mask the input and output, since they will affect the convolution.
        self.enc_c = encoder_block(self.emb_c, num_conv_layers=self.emb_num_convs,
                                   kernel_size=self.emb_kernel_size, hidden_size=self.hidden_size,
                                   num_heads=self.num_heads, num_blocks=self.emb_num_blocks,
                                   mask=self.c_mask, dropout=self.dropout,
                                   scope='encoder_block', reuse=None)
        self.enc_q = encoder_block(self.emb_q, num_conv_layers=self.emb_num_convs,
                                   kernel_size=self.emb_kernel_size,hidden_size=self.hidden_size,
                                   num_heads=self.num_heads, num_blocks=self.emb_num_blocks,
                                   mask=self.q_mask, dropout=self.dropout,
                                   scope='encoder_block', reuse=True)
        self.enc_c = tf.nn.dropout(self.enc_c, 1 - self.dropout)
        self.enc_q = tf.nn.dropout(self.enc_q, 1 - self.dropout)

    def _attention(self):
        sim_mat = trilinear_similarity(self.enc_c, self.enc_q)
        c2q_attn_weights = tf.nn.softmax(mask_logits(sim_mat, mask=tf.expand_dims(self.q_mask, 1)), 2)
        q2c_attn_weights = tf.nn.softmax(mask_logits(sim_mat, mask=tf.expand_dims(self.c_mask, 2)), 1)
        self.c2q = tf.matmul(c2q_attn_weights, self.enc_q)
        self.q2c = tf.matmul(tf.matmul(c2q_attn_weights, q2c_attn_weights, transpose_b=True), self.enc_c)
        self.attn_out = tf.concat([self.enc_c, self.c2q, self.enc_c * self.c2q, self.enc_c * self.q2c], axis=-1)
        self.attn_out = tf.nn.dropout(self.attn_out, 1 - self.dropout)

    def _model_encoder(self):
        self.model_encodes = [
            fc(self.attn_out, self.hidden_size, activation_fn=None, scope='input_projection', reuse=None)
        ]
        for i in range(self.num_enc_layers):
            output = encoder_block(self.model_encodes[i], num_conv_layers=self.enc_num_convs,
                                   kernel_size=self.enc_kernel_size, hidden_size=self.hidden_size,
                                   num_heads=self.num_heads, num_blocks=self.enc_num_blocks, mask=self.c_mask,
                                   dropout=self.dropout, scope='encoder_block', reuse=True if i > 0 else None)
            self.model_encodes.append(tf.nn.dropout(output, 1 - self.dropout))

    def _output(self):
        # TODO: check whether to use the encodes before dropout or after dropout
        self.start_logits = tf.squeeze(
            fc(tf.concat([self.model_encodes[-3], self.model_encodes[-2]], axis=-1),
               1, activation_fn=None, biases_initializer=None, scope='start_pointer'),
            -1)
        self.end_logits = tf.squeeze(
            fc(tf.concat([self.model_encodes[-3], self.model_encodes[-1]], axis=-1),
               1, activation_fn=None, biases_initializer=None, scope='end_pointer'),
            -1)
        self.start_logits = mask_logits(self.start_logits, mask=self.c_mask)
        self.end_logits = mask_logits(self.end_logits, mask=self.c_mask)
        self.start_probs = tf.nn.softmax(self.start_logits)
        self.end_probs = tf.nn.softmax(self.end_logits)

        self.outer_product = tf.matmul(tf.expand_dims(self.start_probs, axis=2),
                                       tf.expand_dims(self.end_probs, axis=1))
        self.outer_product = tf.matrix_band_part(self.outer_product, 0,
                                                 tf.cast(tf.minimum(tf.shape(self.outer_product)[2] - 1,
                                                                    self.max_answer_len), tf.int64))
        self.pred_start = tf.argmax(tf.reduce_max(self.outer_product, axis=2), axis=1)
        self.pred_end = tf.argmax(tf.reduce_max(self.outer_product, axis=1), axis=1)

    def _compute_loss(self):
        self.start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.start_logits, labels=self.start)
        self.end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.end_logits, labels=self.end)
        self.loss = tf.reduce_mean(self.start_loss + self.end_loss, -1)
        with tf.variable_scope('l2_loss'):
            self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.loss += self.weight_decay * self.l2_loss
