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
        
        #iterate through strta, creating test folds
        for _, group in self.df.groupby("strata"):
            #exclude observations that are always_include
            group = group.loc[group["always_include"] == False]
            n_samples = group.shape[0]
            #if not enough observations for the strata, don't add any observations to test_folds
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
                 validation_generator,
                 #desired is very high since if not provided, basically ignored
                 #can be improved
                 desired_time=1e8,
                 patience=10):
        super(TimeStopper, self).__init__()
        self.desired_time = desired_time 
        self.t0 = None
        self.elapsed=0.0
        #if adding optino to monitor diffeent stats, change here
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
        #SHOULD ADD OPTION TO GET LOG LOSS
        #SHOULD HAVE ABILITY TO MONITOR:
        #LOG LOSS, OVERALL ACC, RECALL, PRECISION AND F1
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
#for making a grid and doing a search
def make_grid(args):
        #incase value is not iterable
        iterable_args=collections.defaultdict(list)
        for key, value in args.items():
            if isinstance(value, (tuple,list)):
                iterable_args[key].extend(value)
            else:
                iterable_args[key].append(value)
        #create grid:
        return [dict(zip(iterable_args, x)) for x in itertools.product(*iterable_args.values())]



class NeuralValidator():
    """This class is for testing a certain set of parameters
       Made into a class since will be used with both NeuralSearcher and
       NeuralEvolver
       """
    def __init__(self,
                model_func,
                model_params,
                labels):
        self.model_func = model_func
        self.model_params = model_params
        #this will be a list of dicts
        #each dict will contain latest score for best iteratoin
        self.scores = []
        #labels is passed since NeuralClasser requires it
        #this might be a good reason to split up Tester and Classer
        self.labels = labels

    #always require validation data
    #and use TimeStopper (might want to rename)
    def test(self,
             #this should be a seq of two elments
             train_data,
             valid_data,
             #making maxminutes high so will be nnot a factor if not supplied
             #not ideal, repeating TimeStopper
             max_minutes=1e8,
             class_weights=None):
        classer = core.NeuralClasser(labels = self.labels,
                                        model_func = self.model_func,
                                        **self.model_params)
        validation_generator = core.PadderSequence(x=valid_data[0],
                                                    y=valid_data[1],
                                                    vectorizer=classer.vectorizer,
                                                    prevec=True)
        timestopper = TimeStopper(desired_time = max_minutes,
                                    validation_generator=validation_generator)
        history = classer.fit(x=train_data[0],
                                y=train_data[1],
                                epochs=500,
                                callbacks=[timestopper],
                                class_weights=class_weights)
        scores = dict()
        #now get some stats
        scores['best_epoch'], scores['best_f1'] = max(enumerate(history.history["val_f1"]),
                                                key=operator.itemgetter(1))
        scores['best_time'] = history.history["elapsed"][scores['best_epoch']]
        scores['best_precision'] = history.history["val_precision"][scores['best_epoch']]
        scores['best_recall'] = history.history["val_recall"][scores['best_epoch']]
        #this will always exist in history (unless multiple losses?)
        scores['total_epochs'] = len(history.history["loss"])
        #so that we can later crossreference the individual perclass files with the averaged file
        #WILL NEED TO INTEGRATE INTO SEARCHER
        #scores['best_per_class["model_num"] = params_idx
        self.scores.append(scores)
        #not appending tthe per class to scores, since easier to manage
        return scores, history.history["per_class"][scores['best_epoch']]

class NeuralSearcher():
    #Still need a way to pass arguments to the vectorier
    def __init__(self,
                 #text should be a list of tuples (array or tuple may or may not work)
                 texts,
                 #y is labels, integer encoded (done outside of system....)
                 y,
                 #labels is pandas dataframe with code and label
                 #could get from from y
                 labels,
                 #function that return a keras nodel, compiled
                 model_func,
                 #for iterating over
                 model_param_possibilites={},
                 n_folds=5,
                 train_min=3,
                 always_include=None,
                 class_weights=None,
                 verbose = False):
        
        self.texts=texts
        #don't need to encode here
        self.num_labels = labels.shape[0]
        self.labels=labels
        self.y=y
        self.n_folds=n_folds
        self.train_min=train_min
        self.cv_generator=CustomCV(n_folds, train_min, stratas=self.y, always_include=always_include)
        self.fold_indexes = [(train_indexes, valid_indexes) for train_indexes, valid_indexes in self.cv_generator.split()] 
        self.class_weights=class_weights
        self.model_func = model_func

        #make model arguments sinto grid
        self.param_grid = make_grid(model_param_possibilites)
        self.model_specs = [key for key in model_param_possibilites]
        self.verbose = verbose
        #need to add since starts from 0
        self.results=collections.defaultdict(list)
        self.per_class_results = []
           
       
    #time per fold is maxium number of times that wil be spent on fold, before earlystopping is done
    def fit(self, minutes_per_fold=10, patience=25):
        #iterate through model params
        for params_idx, params in enumerate(self.param_grid):
            #might be good to print parameters.
            print("\n"+"*"*20)
            print("STARTING MODEL SEARCH {}/{}:".format(params_idx, len(self.param_grid)))
            #print model arguments
            for key, value in params.items():
                print("   {} : {}".format(key, value))
            #iterate through folds:
            for fold_idx, (train_index, valid_index) in enumerate(self.fold_indexes):
                print("\n"+"*"*20+"\n")
                print("Starting fold {}/{}".format(fold_idx+1, self.n_folds))
                #need to call clear session to prevent accumlation in memory
                K.clear_session()
                #use the NeuralValidator wrapper
                tester = NeuralValidator(labels = self.labels,
                                         model_func=self.model_func,
                                         model_params=params)
                scores, best_per_class = tester.test((self.texts[train_index], self.y[train_index]),
                                     (self.texts[valid_index], self.y[valid_index]),
                                     max_minutes=minutes_per_fold*60,
                                     class_weights=self.class_weights)
                #Some stats
                print("\nBest F1 score of {:.2%}".format(scores['best_f1']))
                print("Achieved on epoch {}/{} after {:0.0f} seconds".format(scores['best_epoch']+1,
                                                                             scores['total_epochs'],
                                                                             scores['best_time']))
                print("Corresponding to a recall of: {:.2%}".format(scores['best_recall']))
                print("And to a precision of: {:.2%}".format(scores['best_precision']))

                #now model args save to results dict that we will output as DataFrame later
                for key, value in params.items():
                    self.results[key].append(value)
                #saving scores, iterate and append
                for key, value in scores.items():
                    self.results[key].append(value)
                self.results["model_num"].append(params_idx)
                #add model num to per class results so can keep track
                best_per_class["model_num"] = params_idx
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
        #need to convert to string since otherwise None types are silently dropped
        results = results.astype(dtype={col:str for col in self.model_specs})
        model_args = self.model_specs + ["model_num"]
        results = results.groupby(model_args).mean()
        results.to_csv(os.path.join(out_dir,"averaged_results.csv"))
        #now per class results
        perclass_results = pd.concat(self.per_class_results).groupby(["class_int", "model_num"],
                                     as_index=False).mean()
        perclass_results = pd.merge(perclass_results,
                                    self.labels,
                                    left_on = "class_int",
                                    right_index=True)
        for i in range(len(self.param_grid)):
            perclass_result = perclass_results[perclass_results["model_num"] == i]
            perclass_result.to_csv(os.path.join(out_dir,"perclass_result_" + str(i) + ".csv"), index=False)
			
class NeuralEvolver():
    def __init__(self,
                texts,
                y,
                labels,
                model_func,
                possible_params,
                train_min=3,
                always_include=None,
                n_folds=10,
                population_per_fold=20,
                retain=0.4,
                random_select=0.1,
                mutate_chance=0.10,
                class_weights=None):
        """
        Should consider moving out the stuff lik texts, y
        Maybe the creation of cv to fit method
        """
        self.texts = texts

        self.y = y
        self.n_folds=n_folds
        self.train_min=train_min
        self.cv_generator=CustomCV(n_folds, train_min, stratas=self.y, always_include=always_include)
        self.fold_indexes = [(train_indexes, valid_indexes) for train_indexes, valid_indexes in self.cv_generator.split()] 

        #labels are needed since train.NeuralValidator requires it
        #not going to use it, should break dependacy 
        self.labels = labels

        self.class_weights=class_weights
        self.model_func = model_func
        #make certain that value are all iterable (list or tuple)

        self.possible_params = collections.defaultdict(list)
        for key, value in possible_params.items():
            if isinstance(value, (tuple,list)):
                self.possible_params[key].extend(value)
            else:
                self.possible_params[key].append(value)
        self.retain = retain
        self.random_select = random_select
        self.mutate_chance = mutate_chance
        self.population_per_fold = population_per_fold

    def __create_pioneers__(self):
        """Create first generation of network params
           Only needs to be called once, at start of fit

           Returns:
                pop (list): list containing networks params
                            network params are dicts, with two keys

        "Return li of dictionaruies with model args"""

        pop = []
        for _ in range(self.population_per_fold):
            # Create a set of network parameters
            model_params = {}
            for param, values in self.possible_params.items():
                model_params[param] = random.choice(values)
            # Add the network parameters to our population.
            tester = NeuralValidator(labels = self.labels,
                                    model_func=self.model_func,
                                    model_params=model_params)
            pop.append(tester)

        return pop

    def __breed__(self, mother, father):
        """Make two children as parts of their parents.

        Args:
            mother (NeuralValidator): A wrapper for testing a model with certain set of parameters
            father (NeuralValidator): A wrapper for testing a model with certain set of parameters

        Returns:
            children (list): Two instantiated NeuralValidator

        """
        children = []
        for _ in range(2):
            child_params = {}
            # Loop through the parameters and pick params for the kid.
            for param, values in self.possible_params.items():
                #Randomly mutate some of the children's "genes"
                if self.mutate_chance > random.random():
                    child_params[param] = random.choice(
                        values
                    )
                else:
                    child_params[param] = random.choice(
                        [mother.model_params[param], father.model_params[param]]
                    )
            child = NeuralValidator(labels = self.labels,
                                          model_func=self.model_func,
                                          model_params=child_params)
            children.append(child)
        return children

    def fit(self):
    #there's really no arguments to provide
    #that is bad, should move some from init
    #this does the heavy lifting, it evolves the sepcdied number of generations
    
        #start with first gen
        generation = self.__create_pioneers__()
        for fold_idx, (train_index, valid_index) in enumerate(self.fold_indexes):
            print("\n"+"*"*20+"\n")
            print("Starting fold (generation) {}/{}".format(fold_idx+1, self.n_folds))

            #now train...
            #good thing about NeuralValidator is that it does not store weights
            #but still need to clear session
            for network_idx, network in enumerate(generation):
                print("\n"+"*"*20)
                print("STARTING MODEL SEARCH {}/{}:".format(network_idx+1, self.population_per_fold))
                #print model arguments
                for key, value in network.model_params.items():
                    print("   {} : {}".format(key, value))
                _ = network.test((self.texts[train_index], self.y[train_index]),
                                (self.texts[valid_index], self.y[valid_index]),
                                class_weights=self.class_weights)
                K.clear_session()
            #don't evolve if last fold (generation)
            if (fold_idx + 1) < len(self.fold_indexes):
                generation = self.__evolve__(generation)
            
    def __evolve__(self, generation):
        #now sort networks based on score (best_f1)
        graded = [x for x in sorted(generation, key=lambda x: x.scores[-1]["best_f1"], reverse=True)]
        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded)*self.retain)
        # The parents are every network we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(generation) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.__breed__(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)
        return parents