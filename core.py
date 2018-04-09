# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 10:07:38 2018
@author: Ross Beck-MacNeil

This module contains functions and classes that are useful at both train and
test time.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import strip_accents_ascii
import scipy
import keras
from keras import layers
from keras import backend as K
import tensorflow as tf



#this function creates character ngrams from a string
#it incorporates some cleaning: all letters converted to lowercase and the stripping of non-alphanumeric character
#accented characters are kept, for now...
# I would like to convert accented characters to their unaccented counterparts
# I realize that this would be difficult to implement for many languages, but for common french accents, pretty easy
def preprocess_string(string, strip_accents=True):
    string=string.lower()
    if strip_accents:
        string=strip_accents_ascii(string)
    pattern=re.compile('[^a-z0-9]+',re.UNICODE)
    string = pattern.sub(' ', string)
    return string

class TupleTokenizer(object):
    def __init__(self, char_ngrams=(2,4), word_ngrams=(1,1)):

        if char_ngrams is None and word_ngrams is None:
            raise ValueError("At least one of char_ngrams or word_ngrams must be a tuple.")

        self.char_ngrams = char_ngrams
        self.word_ngrams = word_ngrams
    def __call__(self, strings):
        tokens = []
        for string in strings:
            string = preprocess_string(string)
            if self.char_ngrams is not None:
                tokens.extend(create_char_ngrams(string, self.char_ngrams))
            if self.word_ngrams is not None:
                tokens.extend(create_word_ngrams(string, self.word_ngrams))
        return tokens

def create_word_ngrams(string, ngram_range=(1,2)):
    token_pattern=r"(?u)\b\w\w+\b"
    token_pattern = re.compile(token_pattern)
    tokens = token_pattern.findall(string)

    min_n = ngram_range[0]
    max_n = ngram_range[1]

    if max_n != 1:
        original_tokens = tokens
        if min_n == 1:
            tokens = list(original_tokens)
            min_n += 1
        else:
            tokens = []

        n_original_tokens = len(original_tokens)

        tokens_append = tokens.append
        hyph_join ="_".join

        for n in range(min_n, min(max_n+1, n_original_tokens+1)):
            for i in range(n_original_tokens - n + 1):
                tokens_append(hyph_join(original_tokens[i: i + n]))

    return tokens

def create_char_ngrams(string, ngram_range=(2,5)):
    ngrams = []
    white_spaces = re.compile(r"\s+")
    string = white_spaces.sub('', string)
    s_len = len(string)
    for n in range(ngram_range[0], ngram_range[1]+1):
        #offset is start of string, and then increased
        offset = 0
        
        while (offset + n) < s_len :
            offset += 1
            ngrams.append(string[offset:offset + n])
        if offset <= 1: #count a short string (s_len < n) only once
            break
        #only add if length > 0
        ngrams.append(string[:n])
    return ngrams

class Sequencer(object):
    def __init__(self, char_ngrams=(2,4), word_ngrams=(1,1), vocab_size=2**18):

        self.char_ngrams = char_ngrams
        self.word_ngrams = word_ngrams
        self.vocab_size = vocab_size
        self.tokenizer = TupleTokenizer(self.char_ngrams, self.char_ngrams)
        self.vectorizer = HashingVectorizer(tokenizer=self.tokenizer, norm=None, alternate_sign=False,
                                            lowercase=False, n_features=self.vocab_size)
    def __call__(self, tupled_inputs):
        weights = self.vectorizer.transform(tupled_inputs)
        #incrmeenting cols  by one since tensorflow doesn't play well with sparse tensors that have 0 for data vlaue
        shape = (weights.shape[0], weights.shape[1]+1)
        weights = scipy.sparse.csr_matrix((weights.data, weights.indices+1, weights.indptr), shape=shape)
        rows, cols = weights.nonzero()
        ids = scipy.sparse.csr_matrix((cols, (rows, cols)), shape=weights.shape)
        return ids, weights

    
class NeuralContainer(object):
    def __init__(self, labels, char_ngrams=(2,4), word_ngrams=(1,1), vocab_size=2**18):
        self.char_ngrams = char_ngrams
        self.word_ngrams = word_ngrams
        self.vocab_size = vocab_size
        self.sequencer = Sequencer(self.char_ngrams, self.word_ngrams, self.vocab_size)

        #labels shoud be a DataFrame with two columns: code and label
        #should be unique
        self.labels = labels
        self.num_labels = self.labels.shape[0]
        #transform
        self.labels["prediction_idx"] = np.arange(self.num_labels)
        self.labels = self.labels.set_index("code")

    def sequence(self, texts):
        return self.sequencer(texts)

    def encode_labels(self, targets):
        return targets.map(self.labels["prediction_idx"]).values

    def create_model(self, hidden_units=100, non_linear=False, combiner="sum",
	                 lr=0.002, dropout=0.1, embeddings_regularizer=1e-6,
					 prob_regularizer=0.0, **kwargs):
        args = locals()
        
        input_ids = layers.Input(shape=(self.vocab_size+1,), sparse=True, dtype="int64", name="input_ids")
        input_weights = layers.Input(shape=(self.vocab_size+1,), sparse=True, dtype="float32", name="input_weights")
        
        #need to add one, since this is also done by sequencer
        embedded = SparseEmbedding(vocab_size=self.vocab_size+1,
                                   embedding_size=hidden_units,
                                   embeddings_regularizer=keras.regularizers.l2(embeddings_regularizer),
								   combiner=combiner,
                                   name="embedding")([input_ids, input_weights])

        #some regularization in the form of dropout, if greater than 0
        embedded = layers.Dropout(dropout)(embedded)
        
        if non_linear:
            embedded = layers.Dense(units=hidden_units, activation="tanh", name="non_linear")(embedded)
            embedded = layers.Dropout(dropout)(embedded)

        probabilities = layers.Dense(units=self.num_labels,
		                             activation="softmax",
									 kernel_regularizer=prob_regularizer,
                                     name="probabilities")(embedded)

        classify_model = keras.Model(inputs=[input_ids, input_weights], outputs=probabilities)
        classify_model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                               optimizer=keras.optimizers.Nadam(lr),
                               metrics=["acc"])
        return classify_model, args

    def import_model(self, path):
        model = keras.models.load_model(path, custom_objects={"SparseEmbedding":SparseEmbedding})
        #now check if model is compatible with vocab size and seq length
        if model.get_layer("embedding").vocab_size != self.vocab_size +1:
            raise ValueError("Mismatch between sequencer and imported model's vocab size")
        elif model.get_layer("probabilities").units != self.num_labels:
            raise ValueError("Mismatch between container labels and imported model's output length")
        return model

    def create_encoder(self, model):
        return keras.Model(inputs=model.input, outputs=model.get_layer("encoded_products").output)

    def top_classes(self, probabilities, n=3):
        sorted_probs_indexes = np.argsort(-probabilities)
        sorted_probs = -np.sort(-probabilities)

        results = {}
        for i in range(n):
            results["Pred_Code" + str(i)] = self.labels.index.values[sorted_probs_indexes[:,i]]
            results["Pred_Label" + str(i)] = self.labels["label"].values[sorted_probs_indexes[:,i]]
            results["Pred_Prob" + str(i)] = sorted_probs[:,i]
        return pd.DataFrame(results)
    
#custom Keras layer fo average embedding using weights
class WeigtedAverage(layers.Layer):
    def __init__(self, **kwargs):
        super(WeigtedAverage, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        #ouput will have shape of batch_size * embedding_size
        return (input_shape[0][0], input_shape[0][2])

    def compute_mask(self, inputs, mask=None):
        #don't pass mask on
        return None
    def call(self, inputs, mask=None):
        #Assuming that weights are second in tuple...
        #Assumse that weight shave dimesnion: batch_size * seq_length
        #add new axis

        #need to add epsilon to make certain that we never divide by 0à
        #shoiuld look at jkeras code and see how they handle this
        #do they use these sums? or batch_ddot?
        avg = K.sum(inputs[0] * inputs[1], axis=1) / (K.sum(inputs[1], axis=1) + K.epsilon())

        return avg

class SparseEmbedding(layers.Layer):
    """This layer takes two 2d SparseTensor as inputs
    and returns a dense 2d Tensor. It does this by embedding
    and then combining the first SparseTensor, using the second as weights.
    """
    def __init__(self, vocab_size, embedding_size,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 input_length=None,
                 combiner = "sum",
                 **kwargs):
        super(SparseEmbedding, self).__init__(**kwargs)

        if combiner not in ["sum", "mean", "sqrtn"]:
            ValueError('"combiner" must be one of "sum", "mean", "sqrtn"')
        
        self.combiner = combiner
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = keras.constraints.get(embeddings_constraint)
        self.input_length = input_length

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.vocab_size, self.embedding_size),
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint)
        self.built = True

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][:-1] + (self.embedding_size,)
        return output_shape
    
    def call(self, inputs):
        #inputs should be a tuple, with first element is ids and second is weights
        out = tf.nn.embedding_lookup_sparse(self.embeddings, inputs[0], inputs[1], combiner=self.combiner)
        return out

    def get_config(self):
        config = {'vocab_size': self.vocab_size,
                  'embedding_size': self.embedding_size,
                  'embeddings_initializer': keras.initializers.serialize(self.embeddings_initializer),
                  'embeddings_regularizer': keras.regularizers.serialize(self.embeddings_regularizer),
                  'embeddings_constraint': keras.constraints.serialize(self.embeddings_constraint)}
        base_config = super(SparseEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))