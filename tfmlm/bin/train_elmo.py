
import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset
import json
from tensorflow.python.client import device_lib
import os
import tensorflow as tf

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def try_load_options_latest_checkpoint(tf_save_dir):
    options_file = os.path.join(tf_save_dir, 'options.json')
    ckpt_file = tf.train.latest_checkpoint(tf_save_dir)

    try:
        with open(options_file, 'r') as fin:
            options = json.load(fin)
    except:
        options = None

    return options, ckpt_file

def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)

    # define the options
    batch_size = args.batch_size or 128 # batch size for each GPU
    n_gpus = args.n_gpus or 0 

    gpu_list = get_available_gpus()
    if n_gpus <= 0:
        n_gpus = len(gpu_list)
    else:
        n_gpus = min([n_gpus, len(gpu_list)])
    
    print('Work on %s GPUs' % n_gpus)

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = args.n_train_tokens or 768648884

    options = {
     'bidirectional': True,

     'char_cnn': {'activation': 'relu',
      'embedding': {'dim': 16},
      'filters': [[1, 32],
       [2, 32],
       [3, 64],
       [4, 128],
       [5, 256],
       [6, 512],
       [7, 1024]],
      'max_characters_per_token': 50,
      'n_characters': 261,
      'n_highway': 2},
    
     'dropout': 0.1,
    
     'lstm': {
      'cell_clip': 3,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': 512,
      'use_skip_connections': True},

     'transformer': {
       'num_decoder_layers': 8,
       'layer_preprocess': 'layer_norm',
       'hidden_size': 512,
       'filter_size': 2048,
       'num_heads': 8,
       'attention_dropout': 0.1,
       'residual_dropout': 0.1,
       'relu_dropout': 0.1,
       'max_relative_dist': 16,
       'no_additional_dropout': True},

     'use_transformer': True,
     'num_context_steps': 64,
    
     'all_clip_norm_val': 10.0,
     'scale_embeddings': False,
    
     'n_epochs': 10,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 256,
     'n_negative_samples_batch': 8192,
    }

    if args.option_file is not None:
        with open(args.option_file, 'r', encoding='utf-8') as reader:
            options = json.load(reader)
        options['n_train_tokens'] = n_train_tokens
        options['batch_size'] = batch_size
        options['n_tokens_vocab'] = vocab.size
    
    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    checkpoint = None
    if args.load_checkpoint:
        saved_options, checkpoint = try_load_options_latest_checkpoint(args.save_dir)
        if saved_options is not None:
            options = saved_options
            options['batch_size'] = batch_size
    
    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=options.get('shuffle_training_data', True))
    
    validation = None
    if args.valid_prefix is not None:
        validation = BidirectionalLMDataset(args.valid_prefix, vocab, test=True,
                                      shuffle_on_load=False)

    train(options, data, n_gpus, tf_save_dir, tf_log_dir, checkpoint, validation=validation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--valid_prefix', help='Prefix for validation files; Optional')
    parser.add_argument('--n_gpus', type=int, default=0,
                        help='Number of gpus to use. 0 to use all gpus. Default 0.')
    parser.add_argument('--batch_size', type=int, help='batch size for training')
    parser.add_argument('--n_train_tokens', type=int, help='num train tokens')
    parser.add_argument('--load_checkpoint', action='store_true',
                        help='If set, will load latest checkpoint in save_dir if exists.')
    parser.add_argument('--option_file', help='Option file; use default option if empty')

    args = parser.parse_args()
    main(args)

