# -*- coding: utf-8 -*-
"""This module contiains code that is useful for training and evaulating
the perfomance of a model.
"""
import operator
import collections
import itertools
import numpy as np
import pandas as pd
import time
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import BaseCrossValidator
from keras.callbacks import Callback
from math import ceil

from .core import NeuralContainer

import keras.backend as K


class CustomCV(BaseCrossValidator):
    def __init__(self, n_groups, train_min, stratas, always_include=None):
        #N_GROUPS IS number of folds to make...
        self.n_groups = n_groups

        #min number of observations per strata needed for train
        #if there are less than this, inlcude all in train...
        self.train_min = train_min
        
        #make into pandas dataframe.
        #should have option to just start like this..
        #always_include boolean array that indicates observations to always include in the train set
        #useful when updating the model with new retailers, but should not be necessary to provide...
        
        #if always_include not provide, then make 0s (false)
        if always_include is None:
            always_include=np.zeros(stratas.shape[0], dtype=bool)
        
        self.df=pd.DataFrame({"strata":stratas, "always_include":always_include})
    
    def split(self, X=None, y=None, groups=None):       
        #initiliaze empty folds
        #will put indexes that indicate the observation should be in train here
        fold_groups = [list() for i in range(self.n_groups)]
        #indexes = np.arange(0,self.df.shape[0])
        
        #iterate through strta
        for name, group in self.df.groupby("strata"):
            #exclude observations that are always_include
            group = group.loc[group["always_include"] == False]
            n_samples = group.shape[0]
            #if not enough observations for the strata, don't add any them to test_folds
            
            if (n_samples - ceil(n_samples / self.n_groups)) < self.train_min:
                continue
            else:
                #sort to make certain shortest fold gets most
                fold_groups.sort(key = lambda x: len(x))
                num_per_fold = np.ones(self.n_groups, dtype = 'int16')
                num_per_fold = num_per_fold * (n_samples // self.n_groups)
                
                #extras go the beginning
                num_per_fold[:(n_samples % self.n_groups)] += 1
                idxs = np.random.permutation(group.index.values)
                start = 0
                for i, fold_length in enumerate(num_per_fold):
                    fold_groups[i].extend(idxs[start:start+fold_length])
                    start += fold_length
                    
        for fold in fold_groups:
            test_index = np.array(fold)
            train_index = self.df.index.values[~self.df.index.isin(fold)]
            yield train_index, test_index
            
    def get_n_splits(self, X, y=None, groups = None):
        return(self.n_groups)

class TimeStopper(Callback):
    #does early stopping baswed on time and f1
    def __init__(self, desired_time, validation_data=None, patience=10):
        super(TimeStopper, self).__init__()
        self.desired_time = desired_time 
        self.t0 = None
        self.elapsed=0.0
        self.produce_stats = False
        self.best_f1 = 0.
        self.since_best = 0
        self.patience = patience
        
        if validation_data is not None:
            self.produce_stats = True
            self.x = validation_data[0]
            self.y = validation_data[1]
            self.labels = np.unique(self.y)
        
    def on_train_begin(self, logs=None):
        self.t0 = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        self.elapsed = time.time()-self.t0
        #add to logs
        logs["elapsed"] = self.elapsed
        
        if self.produce_stats:
            probs = self.model.predict(self.x , batch_size=512)
            predicted = np.argmax(probs, axis=-1)
            precisions, recalls, f1_scores, supports = precision_recall_fscore_support(self.y, predicted,
                                                                                       average=None, labels=self.labels)
            
            #make into DataFrame
            per_class_stats = pd.DataFrame({"class_int":self.labels, "recall":recalls, "precision":precisions,
                                           "f1_score":f1_scores, "support":supports})
            #cal overall f1 score
            f1_val = np.mean(f1_scores)
            #add to logs
            logs["per_class"] = per_class_stats
            logs["f1_val"] = f1_val
            #and print
            print(" - f1_val {:.2%}".format(f1_val))
    
        if f1_val > self.best_f1:
            self.since_best = 0
            self.best_f1 = f1_val
        else:
            self.since_best += 1
        
        #trigger early stopping
        if (self.elapsed >= self.desired_time) or (self.since_best >= self.patience):
            self.model.stop_training = True
            
class NNGridSearch():
    #will need to modify once have multiple ys
    def __init__(self, text, y, labels, n_folds=5, train_min=3, always_include=None, class_weights=None,
                 verbose = False, model_args={}):
        self.text=text
        
        #y should be a dictionary
        self.neural_helpers = NeuralContainer(labels)
        self.y = self.neural_helpers.encode_labels(y)
        self.n_folds=n_folds
        self.train_min=train_min
        self.cv_generator=CustomCV(n_folds, train_min, stratas=self.y, always_include=always_include)
        self.fold_indexes = [(train_indexes, valid_indexes) for train_indexes, valid_indexes in self.cv_generator.split()] 
        self.class_weights=class_weights
        
        self.verbose = verbose
        
        #SHOULD MAKE model_args prepoulated with defaults
        #AND AOVERIDE iF NECESSARY
        #WOULD MKAE HINTS EAISER
        #will populate from model_args
        params=collections.defaultdict(list)
        #try and escept in case not iterable
        for key, value in model_args.items():
            if isinstance(value, (tuple,list)):
                params[key].extend(value)
            else:
                params[key].append(value)

        #create grid:
        self.grid_params=[dict(zip(params, x)) for x in itertools.product(*params.values())]
        
        self.num_searches = len(self.grid_params)

        #need to add since starts from 0
        self.results=collections.defaultdict(list)
        
        self.per_class_results = []
        
        #self.history = []
           
    def _fit_model(self, model, batch_size=2500, lr=0.02, **kwargs):
        history = model.fit(x=self.x_train, y=self.y_train, validation_data=self.validation_data, epochs=500,
                            class_weight=self.class_weights, verbose=self.verbose, callbacks=[self.time_stopper],
                            batch_size=batch_size)
        return history
        
    
    #time per fold is maxium number of times that wil be spent on fold, before earlystopping is done
    def fit(self, minutes_per_fold=10, patience=25):
        #now iterate through model parameters...
        for i, params in enumerate(self.grid_params):
            #might be good to print parameters.
            print("\n"+"*"*20)
            print("STARTING SEARCH {}/{}:".format(i+1, self.num_searches))
            for key, value in params.items():
                print("   {} : {}".format(key, value))
            
            #first, create inputs
            #self.x is a tuple. First element is the seqs, second is weights for words.
            ids, weights = self.neural_helpers.sequence(self.text)
            self.x = {"input_ids":ids, "input_weights":weights}
            
            #iterate through folds:
            for j, (train_index, valid_index) in enumerate(self.fold_indexes):
                print("\n"+"*"*20+"\n")
                print("Starting fold {}/{}".format(j+1, self.n_folds))
                #need to call clear session to prevent accumlation in memory
                K.clear_session()
                #create model
                model, args = self.neural_helpers.create_model(**params)
                #for checking defualts and such
                #but don't want self
                del args["self"]
                args = {**args, **params}
                self.x_train = {key: value[train_index] for (key, value) in self.x.items()}
                self.x_valid = {key: value[valid_index] for (key, value) in self.x.items()}
                self.y_train = self.y[train_index]
                self.y_valid = self.y[valid_index]
                
                self.validation_data=(self.x_valid, self.y_valid)
                #return self.validation_data
                self.time_stopper = TimeStopper(minutes_per_fold*60, validation_data=self.validation_data)
                               
                history = self._fit_model(model, **params)
                #self.history.append(history)
                #now get some stats
                #WANT TO ADD CALCULATION OF ACCURACY BY CLASS, AND RETURN IT
                #MIGHT NEED A CUSTOM METRIC.... HAVENT DONE YET
                #using accurayc instead of loss
                #not certain if that is the best approach
                best_epoch, best_f1 = max(enumerate(history.history["f1_val"]), key=operator.itemgetter(1))
                best_time = history.history["elapsed"][best_epoch]
                best_loss = history.history["val_loss"][best_epoch]
                best_acc = history.history["val_acc"][best_epoch]
                total_epochs = len(history.history["loss"])
                
                best_per_class = history.history["per_class"][best_epoch]
    
                #Some stats
                print("\nBest F1 score of {:.1%}".format(best_f1))
                print("Achieved on epoch {}/{} after {:0.0f} seconds".format(best_epoch+1, total_epochs, best_time))
                print("\nCorresponding to an unweighted (unbalanced) accuracy of: {:.1%}".format(best_acc))
                print("And to a weighted (balanced) loss of: {:.4}".format(best_loss))
                #now save to dict
                for key, value in args.items():
                    self.results[key].append(value)
                #some more stats
                self.results["best_f1"].append(best_f1)
                self.results["best_acc"].append(best_acc)
                self.results["best_loss"].append(best_loss)
                self.results["best_epoch"].append(best_epoch)
                self.results["total_epochs"].append(total_epochs)
                self.results["best_elapsed"].append(best_time)
                
                self.per_class_results.append(best_per_class)