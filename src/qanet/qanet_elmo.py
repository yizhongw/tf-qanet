#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 04/24/2018 8:11 PM
import numpy as np
from qanet.qanet import *
from bilm.wrapper import BilmWrapper


class QANet_ELMO(QANet):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.elmo_graph = tf.Graph()
        with self.elmo_graph.as_default(), tf.device('/device:GPU:0'):
            self.elmo = BilmWrapper(
                options_file=config.elmo_option_path, weight_file=config.elmo_weight_path
            )
            sess_config = tf.ConfigProto(allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True
            self.elmo_sess = tf.Session(config=sess_config)
            self.elmo_sess.run(tf.global_variables_initializer())

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
            pad_p_len, pad_q_len = len(batch['context_word_ids'][0]), len(batch['question_word_ids'][0])
            with self.elmo_graph.as_default():
                c_elmo = self.elmo.run(self.elmo_sess, [s['context_words'][:pad_p_len] for s in batch['raw_data']])
                q_elmo = self.elmo.run(self.elmo_sess, [s['question_words'][:pad_q_len] for s in batch['raw_data']])
            feed_dict[self.ce] = np.asarray(c_elmo.data)
            feed_dict[self.qe] = np.asarray(q_elmo.data)

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
        pad_p_len, pad_q_len = len(batch['context_word_ids'][0]), len(batch['question_word_ids'][0])
        with self.elmo_graph.as_default():
            c_elmo = self.elmo.run(self.elmo_sess, [s['context_words'][:pad_p_len] for s in batch['raw_data']])
            q_elmo = self.elmo.run(self.elmo_sess, [s['question_words'][:pad_q_len] for s in batch['raw_data']])
        feed_dict[self.ce] = np.asarray(c_elmo.data)
        feed_dict[self.qe] = np.asarray(q_elmo.data)

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
        return batch_preds

    def _inputs(self):
        self.c = tf.placeholder(tf.int32, [None, None], 'context')
        self.q = tf.placeholder(tf.int32, [None, None], 'question')
        self.cc = tf.placeholder(tf.int32, [None, None, None], 'context_char')
        self.qc = tf.placeholder(tf.int32, [None, None, None], 'question_char')
        self.ce = tf.placeholder(tf.float32, [None, 3, None, 1024], 'context_elmo')
        self.qe = tf.placeholder(tf.float32, [None, 3, None, 1024], 'question_elmo')
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
                             size=self.hidden_size, dropout=self.dropout, scope='highway', reuse=None)
        self.emb_q = highway(tf.concat([self.emb_q, self.conv_emb_qc], axis=2),
                             size=self.hidden_size, dropout=self.dropout, scope='highway', reuse=True)

        self.elmo_weights = tf.nn.softmax(tf.get_variable('elmo_weights', [3], dtype=tf.float32, trainable=True,
                                                          initializer=tf.constant_initializer(1.0 / 3)))
        # self.elmo_weights_2 = tf.nn.softmax(tf.get_variable('elmo_weights_2', [3], dtype=tf.float32, trainable=True,
        #                                                     initializer=tf.constant_initializer(1.0 / 3)))
        self.scale_para = tf.get_variable('elmo_scale', [1], dtype=tf.float32, trainable=True,
                                          initializer=tf.constant_initializer(0.2))
        # self.scale_para_2 = tf.get_variable('elmo_scale_2', [1], dtype=tf.float32, trainable=True,
        #                                     initializer=tf.constant_initializer(0.2))
        self.elmo_c = self.scale_para * (self.elmo_weights[0] * self.ce[:, 0, :, :] +
                                         self.elmo_weights[1] * self.ce[:, 1, :, :] +
                                         self.elmo_weights[2] * self.ce[:, 2, :, :])
        self.elmo_q = self.scale_para * (self.elmo_weights[0] * self.qe[:, 0, :, :] +
                                         self.elmo_weights[1] * self.qe[:, 1, :, :] +
                                         self.elmo_weights[2] * self.qe[:, 2, :, :])
        self.elmo_c = tf.nn.dropout(self.elmo_c, 1 - 5.0 * self.dropout)
        self.elmo_q = tf.nn.dropout(self.elmo_q, 1 - 5.0 * self.dropout)

        self.emb_c = tf.concat([self.emb_c, self.elmo_c], -1)
        self.emb_q = tf.concat([self.emb_q, self.elmo_q], -1)

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
                                   kernel_size=self.emb_kernel_size, hidden_size=self.hidden_size,
                                   num_heads=self.num_heads, num_blocks=self.emb_num_blocks,
                                   mask=self.q_mask, dropout=self.dropout,
                                   scope='encoder_block', reuse=True)

        self.enc_c = tf.nn.dropout(self.enc_c, 1 - self.dropout)
        self.enc_q = tf.nn.dropout(self.enc_q, 1 - self.dropout)
        self.enc_c = tf.concat([self.enc_c, self.elmo_c], -1)
        self.enc_q = tf.concat([self.enc_q, self.elmo_q], -1)

