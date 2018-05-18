# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 10:07:38 2018
@author: Ross Beck-MacNeil

This module contains functions and classes that are useful both when developing
and using a model.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import strip_accents_ascii
import keras
from keras import layers
from keras import backend as K
import tensorflow as tf
from sklearn.utils.murmurhash import murmurhash3_bytes_s32

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
  
class Vectorizer(object):
    """
    This operates on a list of tuples. The tuples can be of varying length.
    Each element of the tuple should be a string. 
    It returns a list of tuples, where each tuple is of varying length.
    Each element of tuple is an integer, corresponging to hashed.
    """
    def __init__(self,
                 char_ngrams=(2,4),
                 word_ngrams=(1,1),
                 vocab_size=2**18):
        if char_ngrams is None and word_ngrams is None:
            raise ValueError("At least one of char_ngrams or word_ngrams must be a tuple.")

        self.char_ngrams = char_ngrams
        self.word_ngrams = word_ngrams
        self.vocab_size = vocab_size
        
    def __vectorize__(self, doc):
        #should have some logic regarding an empty document
        num_words = len(doc)
        max_tokens = 0
        hashed_tokens = []
        for string in doc:
            string = preprocess_string(string)
            tokens = []
            if self.char_ngrams is not None:
                tokens.extend(create_char_ngrams(string, self.char_ngrams))
            if self.word_ngrams is not None:
                tokens.extend(create_word_ngrams(string, self.word_ngrams))
            max_tokens = max(max_tokens, len(tokens))
            hashed_tokens.append(hash_vocab(tokens, self.vocab_size))
        return hashed_tokens, num_words, max_tokens 
            
    def __call__(self, docs):
        #docs is an object arrary where each element is a list (or tuple) of string
        #returns two arrays:
        #first is array where each element is a list or tuple that contains lists (or tuples) of integers (hashed tokens)
        #second is an integer arrary with two dims, first dim give number of lists, and second give max lenght of list
        num_docs = len(docs)
        x = np.empty(num_docs, dtype=object)
        shapes = np.zeros((num_docs, 2), dtype = np.int32)
        for i, doc in enumerate(docs):
            hashed_tokens, num_words, max_tokens  = self.__vectorize__(doc)
            x[i] = hashed_tokens
            shapes[i,0] = num_words
            shapes[i,1] = max_tokens
        return x, shapes

def hash_vocab(tokens, n_features = 2**18):
    """
    should make into a class, or the caller into a class
    Want to be able to save n_features so can reuse and compare
    """
    idxs = []
    for token in tokens:
        #if string, encode
        #otherwise, should be bytes (or stright int)
        if isinstance(token, str):
            token = token.encode("utf-8")
        #adding 1 since 0 is reserved for padding/empty
        #can change later
        h = murmurhash3_bytes_s32(token, 0) + 1
        idxs.append(abs(h) % n_features)
    return idxs

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

def create_char_ngrams(string, ngram_range=(3,5)):
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

#for random shuffling
class PadderSequence(keras.utils.Sequence):
    def __init__(self, x, y=None, shapes=None, batch_size=128, vectorizer = None):
        self.x = x
        self.y = y
        self.shapes = shapes
        self.batch_size = batch_size
        self.vectorizer = vectorizer
        if shapes is None and vectorizer is None:
            raise ValueError("shapes must be provided if vectorizer is not")
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        if self.vectorizer is not None:
            batch_x, batch_shapes = self.vectorizer(batch_x)
        else:
            batch_shapes = self.shapes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_shape = batch_shapes.max(axis=0).tolist()
        #create empy array
        x = np.zeros([len(batch_x)]+batch_shape, dtype=np.int32)
        for i, doc in enumerate(batch_x):
            for j, el in enumerate(doc):
                num_tokens = len(el)
                x[i,j, 0:num_tokens] = el
        if self.y is None:
            return x
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return x, batch_y
    def on_epoch_end(self):
        shuffled_idxs = np.random.permutation(len(self.x))
        self.x = self.x[shuffled_idxs]
        self.y = self.y[shuffled_idxs]
        self.shapes = self.shapes[shuffled_idxs]
    
class ReducerSum(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReducerSum, self).__init__(**kwargs)
        self.supports_masking = True
        
    def compute_mask(self, inputs, mask=None):
        #will pass a mask only if all entry were masked 
        if len(inputs.shape)  > 2 and mask is not None:
            mask = K.all(mask, axis=-1, keepdims=False)
        else: #don't return mask if not enough dimsions
            return None     
    
    def compute_output_shape(self, input_shape):
        return input_shape[0:-2] + (input_shape[-1],)
    def call(self, inputs, mask = None):
        #only operate on last axis
        if mask is not None:
            mask = K.cast(mask, 'float32')
            mask = K.expand_dims(mask, axis=-1)
            return K.sum(inputs*mask, axis=-2)
        else:
            return K.sum(inputs, axis=-2)
        
class ReducerMean(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReducerMean, self).__init__(**kwargs)
        self.supports_masking = True
        
    def compute_mask(self, inputs, mask=None):
        if len(inputs.shape)  > 2 and mask is not None:
            mask = K.any(mask, axis=-1, keepdims=False)
        else: #don't return mask
            return None     
    
    def compute_output_shape(self, input_shape):
        return input_shape[0:-2] + (input_shape[-1],)
    def call(self, inputs, mask = None):
        #only operate on last axis
        if mask is not None:
            mask = K.cast(mask, 'float32')
            #add axis for broadcasting
            mask = K.expand_dims(mask, axis=-1)
            return K.sum(inputs*mask, axis=-2) / (K.sum(mask, axis=-2) +K.epsilon())
        else:
            return K.mean(inputs, axis=-2)
        
class ReducerMax(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReducerMax, self).__init__(**kwargs)
        self.supports_masking = True
        
    def compute_mask(self, inputs, mask=None):
        if len(inputs.shape)  > 2 and mask is not None:
            mask = K.all(mask, axis=-1, keepdims=False)
        else: #don't return mask
            return None    
    
    def compute_output_shape(self, input_shape):
        return input_shape[0:-2] + (input_shape[-1],)
    def call(self, inputs, mask = None):
        if mask is not None:
            mask = K.cast(mask, 'float32')
            mask = K.expand_dims(mask, axis=-1)
            return K.max(inputs*mask, axis=-2)
        else:
            return K.max(inputs, axis=-2)


class Dropper(keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropper, self).__init__(**kwargs)
        self.supports_masking = True
        self.rate = min(1., max(0., rate))
        self.seed = seed
        self.noise_shape = noise_shape
        
    def compute_mask(self, inputs, mask=None):
        return mask
    def _get_noise_shape(self, inputs):
        #only operate on last axis
        return self.noise_shape if self.noise_shape else K.concatenate([K.shape(inputs)[:-1], K.ones(1, dtype='int32')])
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs, mask = None, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)
            #zero things out
            #want to not zerot hings out of nothing would not be zero
            def dropped_inputs(inputs=inputs, rate=self.rate, seed=self.seed):
                kept_idx = K.greater_equal(K.random_uniform(noise_shape,
                                                            seed=seed), rate)
                kept_idx = K.cast(kept_idx, K.floatx())
                return inputs*kept_idx
            
        
            return K.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs
    
#this layer combines a laye
class Attention(keras.layers.Layer):
    def __init__(self,
                 kernel_initializer=None,
                 bias_initializer=None,
                 **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.kernel_initializer = keras.initializers.get('glorot_uniform')
        self.bias_initializer = keras.initializers.get('zeros')
        
        
    def compute_mask(self, inputs, mask=None):
        if len(inputs.shape)  > 2 and mask is not None:
            mask = K.all(mask, axis=-1, keepdims=False)
        else: #don't return mask
            return None  
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.sigmoid_kernel = self.add_weight(shape=(input_dim, 1),
                                      initializer=self.kernel_initializer,
                                      name='sigmoid_bias',
                                      regularizer=None,
                                      constraint=None)
        self.sigmoid_bias = self.add_weight(shape=(1, ),
                                      initializer=self.bias_initializer,
                                      name='sigmoid_kernel',
                                      regularizer=None,
                                      constraint=None)
        
        self.tanh_kernel = self.add_weight(shape=(input_dim, input_dim),
                                      initializer=self.kernel_initializer,
                                      name='tanh_kernel',
                                      regularizer=None,
                                      constraint=None)
        self.tanh_bias = self.add_weight(shape=(input_dim, ),
                                           initializer=self.bias_initializer,
                                           name='tanh_bias',
                                           regularizer=None,
                                           constraint=None)

        self.input_spec = keras.engine.InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
    
    def compute_output_shape(self, input_shape):
        return input_shape[0:-2] + (input_shape[-1],)

    def call(self, inputs, mask = None):
        #first, tanh
        attention = K.dot(inputs, self.tanh_kernel)
        attention = K.bias_add(attention, self.tanh_bias)
        #now tanh
        attention = keras.activations.tanh(attention)
        
        #now sigmoid attention vector
        attention = K.dot(attention, self.sigmoid_kernel)
        attention = K.bias_add(attention, self.sigmoid_bias)
        attention = keras.activations.sigmoid(attention)

        #if mask there, then apply (zero out masked)
        if mask is not None:
            mask = K.cast(mask, 'float32')
            #add axis for broadcasting
            mask = K.expand_dims(mask, axis=-1)
            attention = attention * mask

        attention = attention / (K.sum(attention, axis=-1, keepdims=True) +K.epsilon())
        return K.sum(inputs*attention, axis=-2) / (K.sum(attention, axis=-2) +K.epsilon())
    
class NeuralContainer(object):
    def __init__(self, labels,
                 char_ngrams=(2,4),
                 word_ngrams=(1,1),
                 vocab_size=2**18):
        self.char_ngrams = char_ngrams
        self.word_ngrams = word_ngrams
        self.vocab_size = vocab_size

        #labels shoud be a DataFrame with two columns: code and label
        #should be unique
        self.labels = labels
        self.num_labels = self.labels.shape[0]
        #transform
        self.labels["prediction_idx"] = np.arange(self.num_labels)
        self.labels = self.labels.set_index("code")

    def encode_labels(self, targets):
        return targets.map(self.labels["prediction_idx"]).values

    def import_model(self, path):
        #use saved model to instantioe, is easier
        model = self.model_func(vocab_size=self.vocab_size, **self.model_args)
        model.load_weights("path")
        return model

    def create_model(self, model_func, model_args):
        #model func should return a compiled keras model
        #must take vocab size as argument
        #must have an embedding layer named 'embedding'
        self.model_func = model_func
        self.model_args = model_args
        self.model_summary = []
        model = model_func(vocab_size=self.vocab_size, **self.model_args)
        model.summary(print_fn=lambda x:self.model_summary.append(x + "\n"))
        return model
        
    def create_encoder(self, model):
        return keras.Model(inputs=model.input,
                           #should change to last layer rather than by name
                           outputs=model.get_layer("encoded_products").output)

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