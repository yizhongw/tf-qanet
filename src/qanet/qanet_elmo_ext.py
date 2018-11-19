#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 04/24/2018 8:11 PM
import os
import ujson as json
from qanet.qanet_elmo import *


class QANet_ELMO_Ext(QANet_ELMO):

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

        pred_starts, pred_ends, start_probs, end_probs, no_ans_probs = self.sess.run(
            [self.pred_start, self.pred_end, self.start_probs, self.end_probs, self.no_answer_prob], feed_dict)

        batch_preds = []
        for sample_idx in range(len(batch['raw_data'])):
            sample_info = batch['raw_data'][sample_idx]
            no_ans_prob = no_ans_probs[sample_idx]
            pred_start, pred_end = pred_starts[sample_idx], pred_ends[sample_idx]
            ans_prob = start_probs[sample_idx][pred_start] * end_probs[sample_idx][pred_end]
            if ans_prob >= no_ans_prob:
                if 'context_word_offsets' in sample_info:
                    word_offsets = sample_info['context_word_offsets']
                    answer = sample_info['context'][word_offsets[pred_start][0]: word_offsets[pred_end][1] + 1]
                else:
                    self.logger.warning('No offsets information!')
                    context_words = sample_info['context_words'][1:]
                    answer = ' '.join(context_words[pred_start: pred_end + 1])
            else:
                answer = ''
            batch_preds.append({'pred_answer': answer})
        return batch_preds

    def evaluate(self, eval_batches, print_every_n_steps=None, print_result=False):
        if self.use_ema:
            self.sess.run(self.ema_backup_op)
            self.sess.run(self.ema_assign_op)
        all_preds = {}
        for batch_idx, batch in enumerate(eval_batches):
            if print_every_n_steps and (batch_idx + 1) % print_every_n_steps == 0:
                self.logger.info('Processing batch {}...'.format(batch_idx + 1))
            preds = self._predict_batch(batch)
            for sample, pred_info in zip(batch['raw_data'], preds):
                if print_result:
                    self.logger.info('Question: {}'.format(sample['id']))
                    self.logger.info('Gold answers: {}'.format('###'.join(sample['answers'])))
                    self.logger.info('Pred answer: {}'.format(pred_info['pred_answer']))
                all_preds[sample['id']] = pred_info['pred_answer']
        if self.result_dir is not None:
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
            temp_result_path = os.path.join(self.result_dir, 'dev-eval.json')
        else:
            temp_result_path = 'dev-eval.json'
        with open(temp_result_path, 'w') as fout:
            fout.write(json.dumps(all_preds))
        perf = os.popen('python3.6 evaluate-v2.0.py ../data/squad/dev-v2.0.json {}'.format(temp_result_path)).read()
        perf = json.loads(perf)
        if self.use_ema:
            self.sess.run(self.ema_restore_op)
        return perf, all_preds

    def _embed(self):
        with tf.device('/cpu:0'):
            word_pad_emb = tf.get_variable('word_pad_embedding', shape=(1, self.word_vocab.embed_dim),
                                           initializer=tf.zeros_initializer, trainable=False)
            word_unk_emb = tf.get_variable('word_unk_embedding', shape=(1, self.word_vocab.embed_dim),
                                           initializer=tf.zeros_initializer, trainable=True)
            word_nan_emb = tf.get_variable('word_nan_embedding', shape=(1, self.word_vocab.embed_dim),
                                           initializer=tf.zeros_initializer, trainable=True)
            word_emb_init = tf.constant_initializer(self.word_vocab.embeddings[2:]) \
                if self.word_vocab.embeddings is not None \
                else tf.random_normal_initializer()
            normal_word_embs = tf.get_variable('normal_word_embeddings',
                                               shape=(self.word_vocab.size() - 3, self.word_vocab.embed_dim),
                                               initializer=word_emb_init,
                                               trainable=False)
            self.word_emb_mat = tf.concat([word_pad_emb, word_unk_emb, normal_word_embs, word_nan_emb], 0)
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

        # if <nan> token is at the beginning
        # self.no_answer_prob = self.start_probs[:, 0] * self.end_probs[:, 0]
        # self.start_probs = self.start_probs[:, 1:]
        # self.end_probs = self.end_probs[:, 1:]

        # if <nan> token is in the end
        self.no_answer_prob = tf.gather_nd(self.start_probs, tf.stack([tf.range(self.batch_size), self.c_len - 1], 1)) \
                              * tf.gather_nd(self.start_probs, tf.stack([tf.range(self.batch_size), self.c_len - 1], 1))
        real_c_mask = tf.sequence_mask(self.c_len - 1, self.padded_c_len, dtype=tf.float32)
        self.start_probs = self.start_probs * real_c_mask
        self.end_probs = self.end_probs * real_c_mask

        self.outer_product = tf.matmul(tf.expand_dims(self.start_probs, axis=2),
                                       tf.expand_dims(self.end_probs, axis=1))
        self.outer_product = tf.matrix_band_part(self.outer_product, 0,
                                                 tf.cast(tf.minimum(tf.shape(self.outer_product)[2] - 1,
                                                                    self.max_answer_len), tf.int64))
        self.pred_start = tf.argmax(tf.reduce_max(self.outer_product, axis=2), axis=1)
        self.pred_end = tf.argmax(tf.reduce_max(self.outer_product, axis=1), axis=1)

