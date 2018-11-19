#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-2 下午3:47
import h5py
import json
import numpy as np


class Vocab(object):
    def __init__(self, filename=None, lower=False):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}
        self.use_lowercase = lower

        self.embed_dim = None
        self.embeddings = None

        self.pad_token = '<pad>'
        self.unk_token = '<unk>'

        self.add(self.pad_token)  # <pad> -> 0
        self.add(self.unk_token)  # <unk> -> 1

        if filename is not None:
            self.load_file(filename)

    def size(self):
        return len(self.id2token)

    def load_file(self, filename):
        for line in open(filename, 'r'):
            token = line.rstrip('\n')
            self.add(token)

    def get_id(self, key):
        key = key.lower() if self.use_lowercase else key
        if key in self.token2id:
            return self.token2id[key]
        elif key.lower() in self.token2id:
            return self.token2id[key.lower()]
        else:
            return self.token2id[self.unk_token]

    def get_token(self, idx):
        try:
            return self.id2token[idx]
        except KeyError:
            return self.unk_token

    def add(self, label, cnt=1):
        label = label.lower() if self.use_lowercase else label
        if label in self.token2id:
            idx = self.token2id[label]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = label
            self.token2id[label] = idx
        if cnt > 0:
            if label in self.token_cnt:
                self.token_cnt[label] += cnt
            else:
                self.token_cnt[label] = cnt
        return idx

    def filter_tokens_by_cnt(self, min_cnt):
        tokens_to_keep = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        self.add(self.pad_token, 0)
        self.add(self.unk_token, 0)
        for token in tokens_to_keep:
            self.add(token, cnt=0)

    def load_pretrained_embeddings(self, embedding_path):
        trained_embeddings = {}
        if embedding_path.endswith('.hdf5'):
            with h5py.File(embedding_path, 'r') as fin:
                jstr = fin['vocab'][()].decode('ascii', 'strict')
                vocab = json.loads(jstr)
                embed = np.array(fin['embed'])
                for token_idx, token in enumerate(vocab):
                    if token not in self.token2id:
                        continue
                    trained_embeddings[token] = embed[token_idx]
                    if self.embed_dim is None:
                        self.embed_dim = len(embed[token_idx])
        else:
            with open(embedding_path, 'r') as fin:
                for line in fin:
                    contents = line.strip().split(' ')
                    token = contents[0]
                    values = list(map(float, contents[1:]))
                    if token in self.token2id:
                        trained_embeddings[token] = values
                    else:
                        token = token.lower()
                        if token in self.token2id and token not in trained_embeddings:
                            trained_embeddings[token] = values
                    if self.embed_dim is None:
                        self.embed_dim = len(contents) - 1
        filtered_tokens = trained_embeddings.keys()
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        self.add(self.pad_token, 0)
        self.add(self.unk_token, 0)
        for token in filtered_tokens:
            self.add(token, cnt=0)
        # load embeddings
        self.embeddings = np.zeros([self.size(), self.embed_dim])
        for token in self.token2id.keys():
            if token in trained_embeddings:
                self.embeddings[self.get_id(token)] = trained_embeddings[token]

    def convert_to_ids(self, tokens):
        """Convert tokens to ids, use unk_token if the token is not in vocab."""
        vec = []
        vec += [self.get_id(label) for label in tokens]
        return vec

    def recover_from_ids(self, ids, stop_id=None):
        """Recover tokens from ids"""
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens