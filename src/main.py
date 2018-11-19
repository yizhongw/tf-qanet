#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 04/24/2018 8:10 PM
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.append('../tfmlm')
import pickle
import random
import logging
import ujson as json
import numpy as np
import tensorflow as tf
from config import parse_args
from prepro import create_vocab, prepro
from dataset import SQuAD, SQuAD2
from qanet.qanet import QANet
from qanet.qanet_elmo import QANet_ELMO
from qanet.qanet_ext import QANet_Ext
from qanet.qanet_elmo_ext import QANet_ELMO_Ext


def prepare(args):
    prepro(args)
    create_vocab(args)


def init_data_and_model(args):
    logger.info('Loading data and vocabs...')
    if 'ext' in args.model:
        squad_data = SQuAD2(args)
    else:
        squad_data = SQuAD(args)
    logger.info('Train: {}, Dev: {}, Test: {}'.format(len(squad_data.train_samples),
                                                      len(squad_data.dev_samples), len(squad_data.test_samples)))
    with open(os.path.join(args.vocab_dir, 'word.vocab'), 'rb') as fin:
        word_vocab = pickle.load(fin)
        logger.info('Word vocab lower: {}'.format(word_vocab.use_lowercase))
        if word_vocab.embed_dim != args.word_embed_size:
            word_vocab.embed_dim = args.word_embed_size
            word_vocab.embeddings = None
            logger.warning('Reinitialize the word vocab embeddings to dim {}'.format(args.word_embed_size))
        logger.info('Word vocab size: {}'.format(word_vocab.size()))
    with open(os.path.join(args.vocab_dir, 'char.vocab'), 'rb') as fin:
        char_vocab = pickle.load(fin)
        if char_vocab.embed_dim != args.char_embed_size:
            char_vocab.embed_dim = args.char_embed_size
            char_vocab.embeddings = None
            logger.warning('Reinitialize the char vocab embeddings to dim {}'.format(args.char_embed_size))
        logger.info('Char vocab size: {}'.format(char_vocab.size()))
    squad_data.word_vocab = word_vocab
    squad_data.char_vocab = char_vocab
    logger.info('Initializing the model {}...'.format(args.model))
    if args.model == 'qanet':
        model = QANet(args, word_vocab=word_vocab, char_vocab=char_vocab)
    elif args.model == 'qanet-elmo':
        model = QANet_ELMO(args, word_vocab=word_vocab, char_vocab=char_vocab)
    elif args.model == 'qanet-ext':
        word_vocab.add('<nan>')
        model = QANet_Ext(args, word_vocab=word_vocab, char_vocab=char_vocab)
    elif args.model == 'qanet-elmo-ext':
        word_vocab.add('<nan>')
        model = QANet_ELMO_Ext(args, word_vocab=word_vocab, char_vocab=char_vocab)
    else:
        raise NotImplementedError
    return squad_data, model


def train(args):
    data, model = init_data_and_model(args)
    if args.warm_start and os.path.exists(os.path.join(args.model_dir, 'best.meta')):
        model.restore(model_name='best', restore_dir=args.model_dir)
        logger.info('Restart training from model: best')
        model.eval_in_train(data, args.batch_size)
    logger.info('Training the model...')
    model.train(data, args.epochs, args.batch_size,
                eval_every_n_steps=args.eval_every_n_steps, print_every_n_steps=args.print_every_n_steps,
                augment_ratio=args.augment_data_ratio, word_drop_ratio=args.word_drop_ratio)
    logger.info('Done with model training!')


def evaluate(args):
    data, model = init_data_and_model(args)
    model.restore('best')
    logger.info('Evaluating the model...')
    data.max_context_len, data.max_question_len = 1000, 100
    perf, all_preds = model.evaluate(data.gen_mini_batches(args.batch_size, dev=True, shuffle=False),
                                     print_every_n_steps=args.print_every_n_steps)
    if args.result_dir is not None:
        with open(os.path.join(args.result_dir, 'eval_result.json'), 'w') as fout:
            fout.write(json.dumps({'performance': {}, 'predictions': all_preds}))
    logger.info('Eval performance: {}'.format(perf))


if __name__ == '__main__':
    args = parse_args()

    logger = logging.getLogger('MRC')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if args.log_path is not None:
        log_dir = os.path.dirname(args.log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with the following arguments:')
    for arg in sorted(vars(args)):
        logger.info('{}: {}'.format(arg, getattr(args, arg)))

    if args.gpu is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
