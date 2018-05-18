# -*- coding: utf-8 -*-
"""This module contiains code that is useful for training and evaulating
the perfomance of a model.
"""
import pathlib
import os
from inspect import getsource
import time
import operator
import collections
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import BaseCrossValidator
from keras.callbacks import Callback
from math import ceil
import keras.backend as K

from . import core

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
    def __init__(self,
                 desired_time,
                 validation_generator,
                 patience=10):
        super(TimeStopper, self).__init__()
        self.desired_time = desired_time 
        self.t0 = None
        self.elapsed=0.0
        self.best_f1 = 0.
        self.since_best = 0
        self.patience = patience
        self.validation_generator = validation_generator
        self.labels = np.unique(validation_generator.y)
        
    def on_train_begin(self, logs=None):
        self.t0 = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        self.elapsed = time.time()-self.t0
        #add to logs
        logs["elapsed"] = self.elapsed
        
        #monitor f1 score
        probs = self.model.predict_generator(generator=self.validation_generator,
                                             steps=len(self.validation_generator))
        predicted = np.argmax(probs, axis=-1)
        precisions, recalls, f1_scores, supports = precision_recall_fscore_support(self.validation_generator.y,
                                                                                   predicted,
                                                                                   average=None,
                                                                                   labels=self.labels)
            
        #make into DataFrame
        per_class_stats = pd.DataFrame({"class_int":self.labels, "recall":recalls, "precision":precisions,
                                           "f1_score":f1_scores, "support":supports})
        #cal overall f1 score
        val_f1 = np.mean(f1_scores)
        val_recall = np.mean(recalls)
        val_precision = np.mean(precisions)
        #add to logs
        logs["per_class"] = per_class_stats
        logs["val_f1"] = val_f1
        logs["val_recall"] = val_recall
        logs["val_precision"] = val_precision
        #and print
        print(" - val_f1 {:.2%}".format(val_f1))
        print(" - val_recall {:.2%}".format(val_recall))
        print(" - val_precision {:.2%}".format(val_precision))
    
        if val_f1 > self.best_f1:
            self.since_best = 0
            self.best_f1 = val_f1
        else:
            self.since_best += 1
        
        #trigger early stopping
        if (self.elapsed >= self.desired_time) or (self.since_best >= self.patience):
            self.model.stop_training = True
     
class NNGridSearch():
    #Still need a way to pass arguments to the vectorier
    def __init__(self,
                 #text should be a list of tuples (array or tuple may or may not work)
                 texts,
                 #y is labels, integer encoded
                 y,
                 #labels is pandas dataframe with code and label
                 labels,
                 #function that return a keras nodel, compiled
                 model_func,
                 n_folds=5,
                 train_min=3,
                 always_include=None,
                 class_weights=None,
                 verbose = False,
                 model_args={},
                 vectorizer_args={}):
        
        self.texts=texts
        #y should be a dictionary
        self.neural_helpers = core.NeuralContainer(labels)
        self.y = self.neural_helpers.encode_labels(y)
        self.n_folds=n_folds
        self.train_min=train_min
        self.cv_generator=CustomCV(n_folds, train_min, stratas=self.y, always_include=always_include)
        self.fold_indexes = [(train_indexes, valid_indexes) for train_indexes, valid_indexes in self.cv_generator.split()] 
        self.class_weights=class_weights
        self.model_func = model_func
        #PLANNING ON ADDING THE ABILITY TO ITERATE THROUGH VECTORIZER ARGS
        self.vectorizer_args = vectorizer_args
        
        self.verbose = verbose
        
                
        ##try and escept in case not iterable
        params=collections.defaultdict(list)
        for key, value in model_args.items():
            if isinstance(value, (tuple,list)):
                params[key].extend(value)
            else:
                params[key].append(value)
        #create grid:
        self.grid_params=[dict(zip(params, x)) for x in itertools.product(*params.values())]
        #need to add ecotizer        
        self.num_searches = len(self.grid_params)

        #need to add since starts from 0
        self.results=collections.defaultdict(list)
        
        self.per_class_results = []
        
        #self.history = []
           
       
    #time per fold is maxium number of times that wil be spent on fold, before earlystopping is done
    def fit(self, minutes_per_fold=10, patience=25):
        #now iterate through model parameters...
        #planning on adding aiblity to iterate through vectorizers args
        vectorizer = core.Vectorizer(**self.vectorizer_args)
        self.x, self.shapes = vectorizer(self.texts)
        for i, params in enumerate(self.grid_params):
            #might be good to print parameters.
            print("\n"+"*"*20)
            print("STARTING SEARCH {}/{}:".format(i+1, self.num_searches))
            for key, value in params.items():
                print("   {} : {}".format(key, value))
                       
            #iterate through folds:
            for j, (train_index, valid_index) in enumerate(self.fold_indexes):
                print("\n"+"*"*20+"\n")
                print("Starting fold {}/{}".format(j+1, self.n_folds))
                #need to call clear session to prevent accumlation in memory
                K.clear_session()
                #create model
                #should save a layout of the model somwehre...
                #need to make certain that model func take vocab_size arg
                #and that this is read from vectorizer,
                model = self.model_func(**params, vocab_size=vectorizer.vocab_size)
                x_train = self.x[train_index]
                x_valid = self.x[valid_index]
                shapes_train = self.shapes[train_index]
                shapes_valid = self.shapes[valid_index]
                y_train = self.y[train_index]
                y_valid = self.y[valid_index]
                
                train_generator = core.PadderSequence(x=x_train,
                                                      y=y_train,
                                                      shapes=shapes_train)
                
                validation_generator = core.PadderSequence(x=x_valid,
                                                           y=y_valid,
                                                           shapes=shapes_valid)
                
                #return self.validation_data
                self.time_stopper = TimeStopper(minutes_per_fold*60,
                                                validation_generator=validation_generator)
                               
                history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=len(train_generator),
                                      epochs=500,
                                      class_weight=self.class_weights,
                                      verbose=self.verbose,
                                      callbacks=[self.time_stopper])
                #self.history.append(history)
                #now get some stats
                #WANT TO ADD CALCULATION OF ACCURACY BY CLASS, AND RETURN IT
                #MIGHT NEED A CUSTOM METRIC.... HAVENT DONE YET
                #using accurayc instead of loss
                #not certain if that is the best approach
                best_epoch, best_f1 = max(enumerate(history.history["val_f1"]), key=operator.itemgetter(1))
                best_time = history.history["elapsed"][best_epoch]
                best_precision = history.history["val_precision"][best_epoch]
                best_recall = history.history["val_recall"][best_epoch]
                total_epochs = len(history.history["loss"])
                best_per_class = history.history["per_class"][best_epoch]
                #to keep track
                best_per_class["search_num"] = i
    
                #Some stats
                print("\nBest F1 score of {:.2%}".format(best_f1))
                print("Achieved on epoch {}/{} after {:0.0f} seconds".format(best_epoch+1, total_epochs, best_time))
                print("Corresponding to a recall of: {:.2%}".format(best_recall))
                print("And to a precision of: {:.2%}".format(best_precision))
                #now save to dict
                for key, value in params.items():
                    self.results[key].append(value)
                #some more stats
                self.results["best_f1"].append(best_f1)
                self.results["best_recall"].append(best_recall)
                self.results["best_precision"].append(best_precision)
                self.results["best_epoch"].append(best_epoch)
                self.results["total_epochs"].append(total_epochs)
                self.results["best_elapsed"].append(best_time)
                self.results["search_num"].append(i)
                
                self.per_class_results.append(best_per_class)        
    def save_results(self, results_dir):
        time_finished = time.strftime("%Y%m%d_%Hh%Mm")
        out_dir = results_dir + "search_results_" + time_finished
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        #save model string
        with open(os.path.join(out_dir, "model_string.txt"),"w") as f:
            f.write(getsource(self.model_func))
        #now the averaged results per set pof params
        results = pd.DataFrame(self.results)
        params = [key for key in self.grid_params[0].keys()] + ["search_num"]
        results= results.groupby(params).mean()
        results.to_csv(os.path.join(out_dir,"averaged_results.csv"))
        #now per class results
        perclass_results = pd.concat(self.per_class_results).groupby(["class_int", "search_num"], as_index=False).mean()
        perclass_results["class_code"] = self.neural_helpers.labels.index[perclass_results.class_int].values
        perclass_results["class_label"] = self.neural_helpers.labels["label"][perclass_results.class_int].values
        for i in range(self.num_searches):
            perclass_result = perclass_results[perclass_results["search_num"] == i]
            perclass_result.to_csv(os.path.join(out_dir,"perclass_result" + str(i) + ".csv"), index=False)