# -*- coding: utf-8 -*-
"""
This module contains functions and classes which help develop,
train and deploy machine learning models based on Tensorflow
that classify scanner and webscraped data to the CPI product classification.

@author: Ross Beck-MacNeil
"""

import re
import numpy as np
import keras.backend as K
from keras.models import Model
import keras.models
from keras.layers import Layer, Input, Embedding, Dense, BatchNormalization, Multiply, Reshape, TimeDistributed, Dropout, Masking
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Nadam
from keras.constraints import non_neg
from keras.regularizers import l2
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import strip_accents_ascii

import pandas as pd

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

#this function is a wrapper that applies a supplied tokenizer to all elements in an iterables, returning a concated list
##**kwargs are passed are named arguemnts to the calleable tokenizer
#tokenizer should take a string as first positional argument
def tokenize_iter(strings, char_ngrams=(2,4), word_ngrams=(1,2)):
    tokens = []
    for string in strings:
        string = preprocess_string(string)
        if char_ngrams is not None:
            tokens.extend(create_char_ngrams(string, char_ngrams))
        if word_ngrams is not None:
            tokens.extend(create_word_ngrams(string, word_ngrams))

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
        ngrams.append(string[offset:offset + n])
        while offset + n < s_len :
            offset += 1
            ngrams.append(string[offset:offset + n])
        if offset == 1: #count a short string (s_len < n) only once
            break
    return ngrams

def create_seqs(tupled_inputs, seq_len=500, char_ngrams=(2,4), word_ngrams=(1,1), vocab_size=2**18, **kwargs):
    #don't really need this class
    #would be pretty easy to use hasing myself
    vectorizer = HashingVectorizer(tokenizer=lambda x: tokenize_iter(x, char_ngrams=char_ngrams, word_ngrams=word_ngrams),
                               norm=None, alternate_sign=False, lowercase=False, n_features=vocab_size)
    dtm = vectorizer.transform(tupled_inputs)

    seqs = np.zeros((dtm.shape[0], seq_len), dtype=np.int32)
    weights = np.zeros((dtm.shape[0], seq_len), dtype=np.float)
    for i in range(dtm.shape[0]):
        #do stuff to make certain max_len is respected
        #first, extract indexes
        indxs = np.arange(dtm.indptr[i],dtm.indptr[i+1])
        num_tokens=len(indxs)
        #now extract random indexes, up to max_len
        indxs = indxs[np.random.permutation(min(seq_len,num_tokens))]
        len_index=len(indxs)
        #adding 1, since 0 is reserved for padding
        seqs[i,:len_index]=dtm.indices[indxs]+1
        weights[i,:len_index]=dtm.data[indxs]
    return {"seqs":seqs, "seq_weights":weights}

class Sequencer(object):
    def __init__(self, seq_len=500, char_ngrams=(2,4), word_ngrams=(1,1), vocab_size=2**18):

        self.char_ngrams = char_ngrams
        self.word_ngrams = word_ngrams
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.tokenizer = TupleTokenizer(self.char_ngrams, self.char_ngrams)
        self.vectorizer = HashingVectorizer(tokenizer=self.tokenizer, norm=None, alternate_sign=False,
                                            lowercase=False, n_features=self.vocab_size)
    def __call__(self, tupled_inputs):
        dtm = self.vectorizer.transform(tupled_inputs)
        seqs = np.zeros((dtm.shape[0], self.seq_len), dtype=np.int32)
        seq_wghts = np.zeros((dtm.shape[0], self.seq_len), dtype=np.float)
        for i in range(dtm.shape[0]):
            #do stuff to make certain max_len is respected
            #first, extract indexes
            indxs = np.arange(dtm.indptr[i],dtm.indptr[i+1])
            num_tokens=len(indxs)
            #now extract random indexes, up to max_len
            indxs = indxs[np.random.permutation(min(self.seq_len,num_tokens))]
            len_index=len(indxs)
            #adding 1, since 0 is reserved for padding
            seqs[i,:len_index]=dtm.indices[indxs]+1
            seq_wghts[i,:len_index]=dtm.data[indxs]
        return seqs, seq_wghts


class NeuralContainer(object):
    def __init__(self, labels, seq_len=500, char_ngrams=(2,4), word_ngrams=(1,1), vocab_size=2**18):
        self.seq_len = seq_len
        self.char_ngrams = char_ngrams
        self.word_ngrams = word_ngrams
        self.vocab_size = vocab_size

        self.sequencer = Sequencer(self.seq_len, self.char_ngrams, self.word_ngrams, self.vocab_size)

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

    def create_model(self, embedding_size=100, batch_norm=True,
                     learned_weights="embedding", lr=0.007, dropout=0.0, l2_reg=1e-6, **kwargs):


        #embedding_reg = (l2_reg * self.num_labels)/ (self.vocab_size * embedding_size)
        #weight_reg = (l2_reg * self.num_labels)/ (self.vocab_size)

        #l2_reg is appplied to weights final dense prediction layer
        #a multiple is applied to the embedding layer, to account for differences in the number of parametts

        #target lengths should be a dictionary with the name of the target as key and number of unique classes as vlaue
        #might want to change if not using BA
        seqs = Input(shape=(self.seq_len,), name ="seqs")
        
        #embeddings will be regularized (lower variance, closer to 0), if embeddings_lambda > 0
        embeddings = Embedding(self.vocab_size+1, embedding_size, input_length=self.seq_len, mask_zero=True,
                               name='embeddings')(seqs)

        input_weights=Input(shape=(self.seq_len,), name ="seq_wghts")

        weights = Reshape([self.seq_len,1])(input_weights)
        #mask, sot that dropout will not drop 0s
        weights = Masking(name="mask")(weights)
        #some regularization in the form of dropout, if greater than 0
        weights = Dropout(dropout)(weights)

        #additional oppourtunity to learn weights
        #kind of like an "attention" meachnism
        #prefer sigmoid
        if learned_weights == "sigmoid":
            learned_weights = TimeDistributed(Dense(1, activation = "sigmoid"),
                                              name="learned_weights")(embeddings)
            weights = Multiply()([weights, learned_weights])
        elif learned_weights == "embedding":
            learned_weights = Embedding(self.vocab_size+1, 1, input_length=self.seq_len, mask_zero=True,
                                        trainable=True, embeddings_constraint=non_neg(), embeddings_initializer="ones",
                                        name='learned_weights')(seqs)
            weights = Multiply()([weights, learned_weights])

        average = WeigtedAverage(name="average")([embeddings,weights])

        #batch norm for faster learning (and regularization), if greater than 0
        #highly recommened
        if batch_norm:
            average = BatchNormalization(name= "norm")(average)

        #some more dropout (if specified)
        average = Dropout(dropout, name="encoded_products")(average)

        #batch_normalization is a good regularizer...
        probabilities = Dense(units=self.num_labels, activation="softmax",
                              kernel_regularizer=l2(l2_reg), name="probabilities")(average)

        classify_model = Model(inputs=[seqs,input_weights], outputs=probabilities)
        classify_model.compile(loss=sparse_categorical_crossentropy, optimizer=Nadam(lr), metrics=["acc"])
        return classify_model

    def import_model(self, path):
        model = keras.models.load_model(path, custom_objects={"WeigtedAverage":WeigtedAverage})
        #now check if model is compatible with vocab size and seq length
        if model.get_layer("embeddings").input_dim != self.vocab_size +1:
            raise ValueError("Mismatch between sequencer and imported model's vocab size")
        elif model.get_layer("embeddings").input_length != self.seq_len:
            raise ValueError("Mismatch between sequencer and imported model's input length")
        elif model.get_layer("probabilities").units != self.num_labels:
            raise ValueError("Mismatch between container labels and imported model's output length")
        return model

    def create_encoder(self, model):
        return Model(inputs=model.input, outputs=model.get_layer("encoded_products").output)

    def top_classes(self, probabilities, n=3):
        sorted_probs_indexes = np.argsort(-probabilities)
        sorted_probs = -np.sort(-probabilities)

        results = {}
        for i in range(n):
            results["Pred_Code" + str(i)] = self.labels.index.values[sorted_probs_indexes[:,i]]
            results["Pred_Label" + str(i)] = self.labels["label"].values[sorted_probs_indexes[:,i]]
            results["Pred_Prob" + str(i)] = sorted_probs[:,i]
        return pd.DataFrame(results)

#want to add method to get hireachy...


#this is a custom Keras layer
#it is bery basic and not very elegant
#but it works
#when load a Keras model, will nee to point to this
class WeigtedAverage(Layer):
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

#this class is a Keras layer for the neural network
# it is called an Average, it implements proper average...
#this means that longer ones are accorded special status
#might also want to incorporate an an attenion style mechanism...
#make a weighted average, where the weights are learned and are based on the embedding
#but that might be weird....
class GlobalAverage(Layer):
    def __init__(self, unit_norm=False, **kwargs):
        super(GlobalAverage, self).__init__(**kwargs)
        self.unit_norm=unit_norm

    def compute_output_shape(self, input_shape):
        #ouput will have shape of batch_size * embedding_size
        return (input_shape[0][0], input_shape[0][2])

    def compute_mask(self, input, mask=None):
        #don't pass mask on
        return None
    def call(self, inputs, mask=None):
        if mask is not None:
            #mask is boolean, cast to float
            mask=K.cast(mask, K.floatx())
            #mask starts off as 2d, with dimdension: batch_size * seq_len
            #want batch_size * embedding_size * seq_len
            #so use keras rpea
            mask = K.repeat(mask, inputs.shape[-1])
            #now permuate dimentsion...
            mask = K.permute_dimensions(mask, [0,2,1])
            #to make the masked balues in inputs equal to 0
            inputs = inputs * mask
        #now need to use tranpsot from tensofr flow..
        #inputs = K.permute_dimensions(mask, [0,2,1])
        #just multipy mask with inputs?
        #return K.batch_flatten(K.batch_dot(inputs, mask))
        avg = K.sum(inputs, axis=1) / K.sum(mask, axis=1)

        if self.unit_norm:
            avg = avg / (K.epsilon()+K.sqrt(K.sum(K.square(avg),
                                            axis=-1, keepdims=True)))
        return avg