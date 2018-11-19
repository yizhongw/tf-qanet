#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 04/24/2018 8:10 PM

import os
import argparse

data_dir = os.path.abspath('../data')

raw_train_file = os.path.join(data_dir, 'squad', 'train-v1.1.json')
raw_dev_file = os.path.join(data_dir, 'squad', 'dev-v1.1.json')
# raw_test_file = os.path.join(data_dir, 'squad', 'dev-v1.1.json')

prepro_train_file = os.path.join(data_dir, 'squad', 'train-v1.1.processed.json')
prepro_dev_file = os.path.join(data_dir, 'squad', 'dev-v1.1.processed.json')
prepro_test_file = os.path.join(data_dir, 'squad', 'dev-v1.1.processed.json')

glove_emb_file = os.path.join(data_dir, 'glove', 'glove.840B.300d.txt')
elmo_dir = os.path.join(data_dir, 'elmo')
vocab_dir = os.path.join(data_dir, 'vocabs')
result_dir = os.path.join(data_dir, 'results')
model_dir = os.path.join(data_dir, 'models')
summary_dir = os.path.join(data_dir, 'summary')


def parse_args():
    parser = argparse.ArgumentParser('MRC System 1.0')
    parser.add_argument('--comment', type=str,
                        help='comment to show the characteristics of the running model in the log')
    parser.add_argument('--prepare', action='store_true',
                        help='preprocess the SQUAD data, create vocabulary, etc.')
    parser.add_argument('--train', action='store_true', help='train the reading comprehension model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate the model')
    parser.add_argument('--gpu', type=str, help='specify cuda visible gpu devices')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam', help='optimizer type')
    train_settings.add_argument('--warm_start', action='store_true',
                                help='start from the checkpoint in the model_dir if there is any checkpoint')
    train_settings.add_argument('--learning_rate', type=float,
                                default=0.001, help='learning rate')
    train_settings.add_argument('--weight_decay', type=float,
                                default=3e-7, help='weight decay')
    train_settings.add_argument('--dropout', type=float,
                                default=0.1, help='dropout rate')
    train_settings.add_argument('--word_drop_ratio', type=float,
                                default=0.0, help='set some words to unk when training')
    train_settings.add_argument('--ema_decay', type=float,
                                default=0.9999, help='exponential moving average decay')
    train_settings.add_argument('--grad_clip', type=float,
                                default=5.0, help='clip gradients to this norm')
    train_settings.add_argument('--batch_size', type=int,
                                default=32, help='batch size')
    train_settings.add_argument('--augment_data_ratio', type=float,
                                default=0.0, help='replace some data with augment data when training')
    train_settings.add_argument('--eval_every_n_steps', type=int,
                                default=0, help='evaluate the model on dev set every n updating steps')
    train_settings.add_argument('--print_every_n_steps', type=int,
                                default=100, help='print the log info every n steps while training and evaluating')
    train_settings.add_argument('--epochs', type=int,
                                default=80, help='train epochs')  # TODO: change this to number steps
    train_settings.add_argument('--seed', type=int, default=123,
                                help='the random seed')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--model', type=str,
                                default='qanet', help='specify which model you want to use')
    model_settings.add_argument('--word_embed_size', type=int, default=300,
                                help='size of the word embeddings; '
                                     'this will not take effect if pre-trained word embedding path is set.')
    model_settings.add_argument('--char_embed_size', type=int, default=64,
                                help='size of the character embeddings; '
                                     'set it to 0 if you don\'t want to use char embedding')
    model_settings.add_argument('--hidden_size', type=int,
                                default=128, help='hidden size')
    model_settings.add_argument('--num_heads', type=int,
                                default=8, help='the number of heads for multi-head attention')
    model_settings.add_argument('--emb_num_blocks', type=int,
                                default=1, help='the number of blocks for model emb layer')
    model_settings.add_argument('--emb_num_convs', type=int,
                                default=4, help='the number of conv layers for model emb layer')
    model_settings.add_argument('--emb_kernel_size', type=int,
                                default=7, help='the number of kernels for model emb layer')
    model_settings.add_argument('--enc_num_blocks', type=int,
                                default=7, help='the number of blocks for model encoder layer')
    model_settings.add_argument('--enc_kernel_size', type=int,
                                default=5, help='the number of kernels for model encoder layer')
    model_settings.add_argument('--num_enc_layers', type=int,
                                default=3, help='the layer of model encoder')
    model_settings.add_argument('--enc_num_convs', type=int,
                                default=2, help='the number of conv layers')
    model_settings.add_argument('--max_context_len', type=int,
                                default=400, help='max length of the context passage')
    model_settings.add_argument('--max_question_len', type=int,
                                default=50, help='max length of the question')
    model_settings.add_argument('--max_answer_len', type=int,
                                default=30, help='max length of the answer')
    model_settings.add_argument('--max_word_len', type=int,
                                default=16, help='max length of a word')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--raw_train_file', default=raw_train_file,
                               help='the path of the raw SQUAD train file, only used in preprocessing')
    path_settings.add_argument('--raw_dev_file', default=raw_dev_file,
                               help='the path of the raw SQUAD dev file, only used in preprocessing')
    path_settings.add_argument('--raw_test_file',
                               help='the path of the raw SQUAD test file, only used in preprocessing')
    path_settings.add_argument('--train_file', default=prepro_train_file,
                               help='the path of the preprocessed train file')
    path_settings.add_argument('--dev_file', default=prepro_dev_file,
                               help='the path of the preprocessed dev file')
    path_settings.add_argument('--augment_files', nargs='+',
                               help='specify the augment data files if you want to use them in training')
    path_settings.add_argument('--feature_files', nargs='+',
                               help='specify the feature files if you want to train any model with features')
    path_settings.add_argument('--test_file',
                               help='the path of the preprocessed test file')
    path_settings.add_argument('--word_embed_path', default=glove_emb_file,
                               help='the path of the pre-trained word embeddings')
    path_settings.add_argument('--char_embed_path',
                               help='the path of the pre-trained char embeddings')
    path_settings.add_argument('--vocab_dir', default=vocab_dir,
                               help='the dir to save vocabularies')
    path_settings.add_argument('--elmo_option_path',
                               default='../data/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',
                               help='the path of the elmo option file')
    path_settings.add_argument('--elmo_weight_path',
                               default='../data/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5',
                               help='the path of the elmo weight file')
    path_settings.add_argument('--model_dir', default=model_dir,
                               help='the dir to save models')
    path_settings.add_argument('--result_dir', default=result_dir,
                               help='the directory to save results')
    path_settings.add_argument('--summary_dir', default=summary_dir,
                               help='the directory to dump tensorflow summary')
    path_settings.add_argument('--log_path', help='the file to output log')
    return parser.parse_args()
