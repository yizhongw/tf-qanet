#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 04/24/2018 8:11 PM

import os
import ujson as json
import time
import logging
import tensorflow as tf
from official_evaluate import exact_match_score, f1_score, metric_max_over_ground_truths


class BaseModel(object):
    def __init__(self, config, **kwargs):
        self.logger = logging.getLogger('MRC')
        # basic config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        self.emb_num_blocks = config.emb_num_blocks
        self.enc_num_blocks = config.enc_num_blocks
        self.emb_kernel_size = config.emb_kernel_size
        self.enc_kernel_size = config.enc_kernel_size
        self.num_enc_layers = config.num_enc_layers
        self.emb_num_convs = config.emb_num_convs
        self.enc_num_convs = config.enc_num_convs

        self.optim_type = config.optim
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.grad_clip = config.grad_clip
        self.dropout_value = config.dropout
        self.use_ema = config.ema_decay > 0
        self.best_perf = None

        # length limit
        self.max_answer_len = config.max_answer_len

        # the vocabs
        self.word_vocab = kwargs['word_vocab']
        self.char_vocab = kwargs['char_vocab']

        # session info
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        start_time = time.time()

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # self.regularizer = tc.layers.l2_regularizer(scale=config.weight_decay)
        self._build_graph()
        self.all_params = tf.trainable_variables()
        # param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        # self.logger.info('There are {} parameters in the model'.format(param_num))

        if self.use_ema:
            self.logger.info('Using Exp Moving Average to train the model with decay {}.'.format(config.ema_decay))
            self.ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.global_step)
            self.ema_op = self.ema.apply(self.all_params)
            with tf.control_dependencies([self.train_op]):
                self.train_op = tf.group(self.ema_op)
            with tf.variable_scope('backup_variables'):
                self.bck_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False,
                                                 initializer=var.initialized_value()) for var in self.all_params]
            self.ema_backup_op = tf.group(*(tf.assign(bck, var.read_value())
                                            for bck, var in zip(self.bck_vars, self.all_params)))
            self.ema_restore_op = tf.group(*(tf.assign(var, bck.read_value())
                                             for bck, var in zip(self.bck_vars, self.all_params)))
            self.ema_assign_op = tf.group(*(tf.assign(var, self.ema.average(var).read_value())
                                            for var in self.all_params))

        # initialize the model
        self.sess.run(tf.global_variables_initializer())
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_time))

        # save info
        self.result_dir = config.result_dir
        if self.result_dir is not None and not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.model_dir = config.model_dir
        if self.result_dir is not None and not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.saver = tf.train.Saver(max_to_keep=50)

    def _build_graph(self):
        raise NotImplementedError

    def _create_optimizer(self, optim_type, learning_rate):
        if optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.8, beta2=0.999, epsilon=1e-7)
        elif optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
        elif optim_type == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        elif optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(optim_type))

    def _get_train_op(self, loss):
        warm_up_lr = tf.minimum(self.learning_rate,
                                self.learning_rate / tf.log(1000.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
        self._create_optimizer(self.optim_type, warm_up_lr)
        # self._create_optimizer(self.optim_type, self.learning_rate)
        grads, vars = zip(*self.optimizer.compute_gradients(loss, colocate_gradients_with_ops=True))
        # grads, vars = zip(*self.optimizer.compute_gradients(loss))
        if self.grad_clip is not None and self.grad_clip > 0:
            grads, grad_norm = tf.clip_by_global_norm(grads, self.grad_clip)
        else:
            grad_norm = tf.global_norm(grads)
        train_op = self.optimizer.apply_gradients(zip(grads, vars), global_step=self.global_step)
        return grads, grad_norm, train_op

    def train(self, dataset, epochs, batch_size,
              eval_every_n_steps=0, print_every_n_steps=0, augment_ratio=0.0, word_drop_ratio=0.0):
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = dataset.gen_mini_batches(batch_size, train=True, shuffle=True,
                                                     augment_ratio=augment_ratio, word_drop_ratio=word_drop_ratio)

            if eval_every_n_steps > 0:
                train_loss = self._train_epoch(train_batches,
                                               lambda step_cnt: self.eval_in_train(dataset, batch_size, None, step_cnt),
                                               eval_every_n_steps, print_every_n_steps)
                self.logger.info('Average train loss: {}'.format(train_loss))
            else:
                train_loss = self._train_epoch(train_batches, print_every_n_steps=print_every_n_steps)
                self.logger.info('Average train loss: {}'.format(train_loss))
                self.eval_in_train(dataset, batch_size, epoch)

    def eval_in_train(self, dataset, batch_size, epoch_cnt=None, step_cnt=None, save_result=True):
        if epoch_cnt is not None:
            self.logger.info('Evaluating the model after {} epoch(s)'.format(epoch_cnt))
        if step_cnt is not None:
            self.logger.info('Evaluating the model after {} batch(es)'.format(step_cnt))
        # relax the length limitations when testing the model
        max_context_len, max_question_len = dataset.max_context_len, dataset.max_question_len
        dataset.max_context_len, dataset.max_question_len = 1000, 100
        eval_batches = dataset.gen_mini_batches(batch_size, dev=True, shuffle=False)
        perf, all_preds = self.evaluate(eval_batches)
        dataset.max_context_len, dataset.max_question_len = max_context_len, max_question_len
        self.logger.info('Dev performance: {}'.format(perf))
        if self.best_perf is None or perf['f1'] > self.best_perf['f1']:
            self.save('best')
            self.best_perf = perf
            if save_result and self.result_dir is not None:
                with open(os.path.join(self.result_dir, 'best_dev_result.json'), 'w') as fout:
                    fout.write(json.dumps({'performance': perf, 'predictions': all_preds}))
        if epoch_cnt is not None and (epoch_cnt < 10 or epoch_cnt % 10 == 0):
            self.save('epoch-{}-f1-{}'.format(epoch_cnt, perf['f1']))
        return perf

    def _train_epoch(self, train_batches, eval_func=None, eval_every_n_steps=None, print_every_n_steps=None):
        raise NotImplementedError

    def evaluate(self, eval_batches, print_every_n_steps=None, print_result=False):
        if self.use_ema:
            self.sess.run(self.ema_backup_op)
            self.sess.run(self.ema_assign_op)
        f1 = exact_match = total = 0
        all_preds = []
        for batch_idx, batch in enumerate(eval_batches):
            if print_every_n_steps and (batch_idx + 1) % print_every_n_steps == 0:
                self.logger.info('Processing batch {}...'.format(batch_idx + 1))
            preds = self._predict_batch(batch)
            for sample, pred_info in zip(batch['raw_data'], preds):
                total += 1
                cur_f1 = metric_max_over_ground_truths(f1_score, pred_info['pred_answer'], sample['answers']) \
                    if len(sample['answers']) > 0 else 0
                cur_em = metric_max_over_ground_truths(exact_match_score, pred_info['pred_answer'], sample['answers']) \
                    if len(sample['answers']) > 0 else 0
                if print_result:
                    self.logger.info('Question: {}'.format(sample['id']))
                    self.logger.info('Gold answers: {}'.format('###'.join(sample['answers'])))
                    self.logger.info('Pred answer: {}'.format(pred_info['pred_answer']))
                # context_words = sample['context_words'][:]
                # context_words[pred_info['start_idx']] = '###' + context_words[pred_info['start_idx']]
                # context_words[pred_info['end_idx']] += '###'
                # context_words[sample['answer_spans'][0][0]] = '%%%' + context_words[sample['answer_spans'][0][0]]
                # context_words[sample['answer_spans'][0][1]] += '%%%'
                all_preds.append(
                    {'uid': sample['id'],
                     # 'question': sample['question'],
                     # 'context': ' '.join(context_words),
                     'pred_answers': [pred_info['pred_answer']],
                     # 'ref_answers': sample['answers'],
                     'f1': cur_f1, 'exact_match': cur_em,
                     'start_probs': pred_info['start_probs'] if 'start_probs' in pred_info else None,
                     'end_probs': pred_info['end_probs'] if 'end_probs' in pred_info else None
                     }
                )
                f1 += cur_f1
                exact_match += cur_em
        perf = {'exact_match': 100.0 * exact_match / total,
                'f1': 100.0 * f1 / total}
        if self.use_ema:
            self.sess.run(self.ema_restore_op)
        return perf, all_preds

    def _predict_batch(self, batch):
        raise NotImplementedError

    def save(self, model_name, save_dir=None):
        if save_dir is None:
            save_dir = self.model_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.saver.save(self.sess, os.path.join(save_dir, model_name))
        self.logger.info('Model saved with name: {}.'.format(model_name))

    def restore(self, model_name, restore_dir=None):
        if restore_dir is None:
            restore_dir = self.model_dir
        self.saver.restore(self.sess, os.path.join(restore_dir, model_name))
        self.logger.info('Model restored from {}'.format(os.path.join(restore_dir, model_name)))
