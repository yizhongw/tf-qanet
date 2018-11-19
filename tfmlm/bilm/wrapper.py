import tensorflow as tf
import json
import numpy as np

from bilm.model import BidirectionalLanguageModel, BidirectionalLanguageModelGraph
from bilm.data import Batcher, TokenBatcher


def _make_precomputed_batch(sentences, embed_table, bos, eos):
    seq_length = [(len(sentence) + 2) for sentence in sentences]
    max_seq_len = max(seq_length)
    batch_size = len(sentences)
    dim = bos.shape[0]
    inputs = np.zeros(dtype=np.float32, shape=[batch_size, max_seq_len, dim])
    for sid, sentence in enumerate(sentences):
        inputs[sid,0] = bos
        inputs[sid,len(sentence) + 1] = eos
        for wid, word in enumerate(sentence):
            if word in embed_table.keys():
                inputs[sid, wid + 1] = embed_table[word]
    
    return inputs, seq_length


class BilmWrapper(object):

    def __init__(
            self,
            options_file: str,
            weight_file: str,
            use_character_inputs=True,
            lm_vocab_file=None,
            embedding_weight_file=None,
            max_batch_size=128,
        ):

        self.use_precomputed_inputs = use_character_inputs
        self.model = BidirectionalLanguageModel(
                                        options_file,
                                        weight_file,
                                        use_character_inputs=use_character_inputs,
                                        embedding_weight_file=embedding_weight_file,
                                        max_batch_size=max_batch_size,
                                        use_precomputed_inputs=self.use_precomputed_inputs)
        if self.use_precomputed_inputs:
            self.char_model = BidirectionalLanguageModel(
                                        options_file,
                                        weight_file,
                                        use_character_inputs=use_character_inputs,
                                        use_precomputed_inputs=False,
                                        max_batch_size=1,
                                        char_part_only=True
            )
        with open(options_file, 'r', encoding='utf-8') as f:
            self.options = json.load(f)
        
        self.use_char = use_character_inputs
        self.max_batch_size = max_batch_size
        if self.use_char:
            max_char_per_token = self.options['char_cnn'].get('max_characters_per_token')
            self.input_ids = tf.placeholder('float32', (None, None, self.options['lstm']['projection_dim']))
            self.char_input_ids = tf.placeholder('int32', (1, None, max_char_per_token))
            self.seq_length = tf.placeholder('int32', (None))
            self.ops = self.model(self.input_ids, self.seq_length)
            self.char_ops = self.char_model(self.char_input_ids)
            self.char_batcher = Batcher(lm_vocab_file, max_char_per_token)
        else:
            self.input_ids = tf.placeholder('int32', (None, None))
            self.ops = self.model(self.input_ids)
            self.batcher = TokenBatcher(lm_vocab_file)

        if self.options.get('use_transformer', False):
            self._num_layers = self.options['transformer'].get('num_decoder_layers') + 1
            self._dim_per_layer = self.options['transformer'].get('hidden_size') * 2
        else:
            self._num_layers = self.options['lstm'].get('n_layers')
            self._dim_per_layer = self.options['lstm'].get('projection_dim')
        
        self.embed_cache = {}
        self.bos = None
        self.eos = None
    
    def num_layers(self):
        return self._num_layers

    def dim_per_layer(self):
        return self._dim_per_layer

    def _compute_char_embedding(self, sess, sentences):
        new_words = set()
        for s in sentences:
            for w in s:
                if w not in self.embed_cache:
                    new_words.add(w)
        if len(new_words) <= 0:
            return
        new_words = list(new_words)
        X = self.char_batcher.batch_sentences([new_words])
        token_embeddings = sess.run(
            self.char_ops['token_embeddings'],
            feed_dict={self.char_input_ids: X}
        )
        if self.bos is None:
            self.bos = token_embeddings[0][0]
        if self.eos is None:
            self.eos = token_embeddings[0][-1]
        for i in range(len(new_words)):
            self.embed_cache[new_words[i]] = token_embeddings[0][i + 1]

    def run(self, sess, sentences):
        """
        This function computes the lm embeddings for the sentences
        :param sess: tensorflow session
        :param sentences: List of sentences.
                      Each sentence is a list of token strings

        :returns:  tensor of float32 of shape
              [num of sentences, num_layers, max sentence length, 2 * dim]
        """
        if len(sentences) > self.max_batch_size:
            raise ValueError("Number of sentences (%d) must be less or equal to "
                            "max_batch_size (%d)." % (len(sentences), self.max_batch_size))
        if not self.use_char:
            X = self.batcher.batch_sentences(sentences)
            lm_embeddings = sess.run(
                self.ops['lm_embeddings'],
                feed_dict={self.input_ids: X}
            )
        else:
            self._compute_char_embedding(sess, sentences)
            embed_input, seq_len = _make_precomputed_batch(sentences, self.embed_cache, self.bos, self.eos)
            lm_embeddings = sess.run(
                self.ops['lm_embeddings'],
                feed_dict={self.input_ids: embed_input, self.seq_length: seq_len}
            )
        return lm_embeddings