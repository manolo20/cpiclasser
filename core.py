# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 10:07:38 2018
@author: Ross Beck-MacNeil

This module contains functions and classes that are useful both when developing
and using a model.
"""
import os
import sys
import re
import pathlib
import pickle
import json
from inspect import getsource
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import strip_accents_ascii
import keras
from keras import layers
from keras import backend as K
import tensorflow as tf
from sklearn.utils.murmurhash import murmurhash3_bytes_s32

#some cleaning: all letters converted to lowercase and the stripping of non-alphanumeric character
#accented characters are kept, for now...
# onvert accented characters to their unaccented counterparts
#difficult to implement for most languages, but for common french accents, pretty easy
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
    Convert character tokens to integers
    based on hashing. Avoids need to see whole data set before assiging.
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
    def __init__(self,
                 x,
                 y=None,
                 shapes=None,
                 batch_size=128,
                 vectorizer = None,
                 prevec=False):
        self.x = x
        self.y = y
        self.shapes = shapes
        self.batch_size = batch_size
        self.vectorizer = vectorizer
        self.prevec = prevec
        if vectorizer is not None and prevec:
            self.x, self.shapes = vectorizer(x)
        if shapes is None and vectorizer is None:
            raise ValueError("shapes must be provided if vectorizer is not")
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        if self.shapes is not None:
            batch_shapes = self.shapes[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_x, batch_shapes = self.vectorizer(batch_x)
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
        if self.shapes is not None:
            self.shapes = self.shapes[shuffled_idxs]
    
class NeuralClasser(object):
    def __init__(self,
                labels,
                model_func,
                batch_size=128,
                char_ngrams=(2,4),
                word_ngrams=(1,1),
                vocab_size=2**18,
                **kwargs):
        #KWARGS ARE PASSED TO MODEL FUNC
        #model func should return a compiled keras model
        #must take vocab size as argument
        #also take num_labels as arg
        #must have an embedding layer named 'embedding'
        #should add a check for embedding...
        self.model_func = model_func
        self.char_ngrams = char_ngrams
        self.word_ngrams = word_ngrams
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        self.model_args = kwargs

        #labels shoud be a DataFrame with at least one column
        #each element of a particular column should be unique
        self.labels = labels.copy()
        self.num_labels = self.labels.shape[0]
        self.labels.reset_index(drop=True, inplace=True)
        #CREATE MODEL HERE
        self.model = self.model_func(vocab_size=self.vocab_size,
                                     num_labels=self.num_labels,
                                     **self.model_args)
        #CREATE ENCODER MODEL HERE
        #MAYBE SAVE MODEL STRING HERE?
        self.model_string = getsource(model_func)

        #CREATE VECTORIZER HERE
        self.vectorizer = Vectorizer(vocab_size=self.vocab_size,
                                     char_ngrams=self.char_ngrams,
                                     word_ngrams=self.word_ngrams)
    def encode_labels(self,
                      targets,
                      src_name="label"):
        #source name should be name of column in 
        return targets.map(pd.Series(self.labels.index.values, index=self.labels[src_name].values)).values


    #NEED TO MAKE A FUNCTION FOR LOADIND MODEl
    #SINCE CAN'T CREATE WITHOUT A MDOEL FUNC
    def import_model(self, path):
        #use saved model to instantioe, is easier
        model = self.model_func(vocab_size=self.vocab_size, **self.model_args)
        model.load_weights("path")
        return model

    #NEED METHODS FOR:
    #SAVING
    #TRAINING
    #PREDICTING
    #AND MAYBE ENCODING
    def fit(self,
            x,
            y,
            epochs=500,
            validation_data=None,
            callbacks=None,
            class_weights=None):
        train_generator = PadderSequence(x=x,
                                         y=y,
                                         vectorizer=self.vectorizer,
                                         batch_size=self.batch_size,
                                         prevec=True)                        
        return self.model.fit_generator(generator=train_generator,
                                      steps_per_epoch=len(train_generator),
                                      epochs=epochs,
                                      class_weight=class_weights,
                                      verbose=1,
                                      callbacks=callbacks)
       
    def predict_top_n(self,
                      texts,
                      concordance=None,
                      n=3):
        """
        Returns: probs, labels
        """

        #text to seqs
        generator = PadderSequence(x=texts,
                                   vectorizer=self.vectorizer,
                                   batch_size=self.batch_size,
                                   prevec=False)  
        #raw prediction probabilities
        probs = self.model.predict_generator(generator=generator,
                                             steps=len(generator),
                                             verbose=1)
        sorted_probs_idxs = np.argsort(-probs)
        sorted_probs = -np.sort(-probs)
        results = {}
        for i in range(n):
            #CAN UPDATE TO INCOROREPATE MULTIPLE LAVEL COLUMNS IN BY LOOPING
            #WOULD HAVE TO EXCLUDE "prediction_idx"
            results["Pred_Label" + str(i)] = self.labels.index.values[sorted_probs_idxs[:,i]]
            results["Pred_Prob" + str(i)] = sorted_probs[:,i]
        return pd.DataFrame(results)

    def save(self,
             out_dir):
        #path should either be an empy directory
        #will be created if it does not exist
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        #first save nice summary
        model_summary = []
        self.model.summary(print_fn=lambda x: model_summary.append(x + "\n"))
        with open(os.path.join(out_dir, "model_summary.txt"), 'w') as f:
            f.write("".join(model_summary))
        #now save parameters
        #first make into dict
        params = {
                "model_args":self.model_args,
                "batch_size":self.batch_size,
                "char_ngrams":self.char_ngrams,
                "word_ngrams":self.word_ngrams,
                "vocab_size":self.vocab_size
        }
        with open(os.path.join(out_dir, "model_params.txt"), 'w') as f:  
            json.dump(params, f, indent=4, separators=(',', ': '))
        #save both model string (so can read) and pickled (so can load)
        with open(os.path.join(out_dir, "model_func.txt"), 'w') as f:  
            f.write(self.model_string)
        with open(os.path.join(out_dir, "model_func.pkl"), 'wb') as f:
            pickle.dump(self.model_func, f)
        #now labels
        self.labels.to_csv(os.path.join(out_dir, 'labels.csv'))
        #finally, model weghts
        self.model.save_weights(os.path.join(out_dir, 'model_weights.h5'))
        
        
    def encode_texts(self,
                     x,
                     batch_size=128,):
        generator = PadderSequence(x=x,
                                   vectorizer=self.vectorizer,
                                   batch_size=self.batch_size)
        encoder = keras.Model(inputs = self.model.input,
                              outputs = self.model.layers[-2].output)
        return encoder.predict_generator(generator=generator,
                                         steps=len(generator),
                                         verbose=1)
def load_classer(in_dir):
    #path should be folder containing necessary files
    with open(os.path.join(in_dir, "model_params.txt")) as f:
        params = json.load(f)
    #now model_func
    with open(os.path.join(in_dir, "model_func.pkl"), "rb") as f:
        model_func = pickle.load(f)
    #now labels
    labels = pd.read_csv(os.path.join(in_dir, "labels.csv"),
                         index_col="label")
    classer = NeuralClasser(labels=labels,
                         model_func=model_func,
                         **params)
    #now wghts
    classer.model.load_weights(os.path.join(in_dir, "model_weights.h5"))
    return classer