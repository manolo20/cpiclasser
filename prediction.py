# -*- coding: utf-8 -*-
import os
import sys
from sklearn.externals import joblib
import numpy as np
import pandas as pd

#add parent folder of file to python path, so can find helpers
#when loading pretrained model
PARENT_DIR = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.insert(0, PARENT_DIR)

   
def predict(df, concordance=None):
    """
    Loads pretrained models and processes raw inputs.
    Returns: probs, labels
    """
    #might want to change it to return probs and labels, with option for 
    texts = [tuple(row) for row in df.values]

    #load models
    classifier = joblib.load(os.path.join(PARENT_DIR, "models", "classifier.pkl"))
    model = classifier.import_model(os.path.join(PARENT_DIR, "models", "neuralnetwork.h5"))
    print("Loaded model.")

    #text to seqs
    seqs, seq_wghts = classifier.sequence(texts)
    x = {"seqs":seqs, "seq_wghts":seq_wghts}
    print("Processed input.")
    
    #raw prediction probabilities
    probs = model.predict(x, batch_size=512, verbose=1)
    
    #if a concordance is supplied, then use it to aggregate
    #if concordnace supplied, use it
    if concordance is not None:
        probs, labels = aggregate_probs(concordance, classifier.labels, probs)
    else:
        labels = classifier.labels
        
    return probs, labels
    

def top_n(probs, labels, n=1):
    """
    A helper function for retrievving labels and codes
    """
    #get top class
    sorted_probs_indexes = np.argsort(-probs)
    sorted_probs = -np.sort(-probs)

    top_classes = {}
    for i in range(n):
        top_classes["Pred_Code" + str(i)] = labels.index.values[sorted_probs_indexes[:,i]]
        top_classes["Pred_Label" + str(i)] = labels["label"].values[sorted_probs_indexes[:,i]]
        top_classes["Pred_Prob" + str(i)] = sorted_probs[:,i]
    top_classes = pd.DataFrame(top_classes)
    
    return top_classes


def aggregate_probs(concordance, model_labels, base_probs):
    #join on indexes, only need preiction_idx column
    concordance = pd.concat([model_labels["prediction_idx"], concordance], axis = 1)
    
    #need to get unique idx for each aggregate
    aggregates = concordance[["code", "label"]].drop_duplicates()
    aggregates["prediction_idx"] = np.arange(aggregates.shape[0])
    
    #should specify prefixes (by default left is x, right is y?)
    concordance = concordance[["code", "prediction_idx"]].merge(aggregates, on="code")
    
    #now create a numpy array as num_EA X num_BAs
    concord_array = np.zeros((concordance.prediction_idx_x.max()+1, concordance.prediction_idx_y.max()+1), dtype=np.float)
    #now set the overlap to ones
    concord_array[concordance["prediction_idx_x"], concordance["prediction_idx_y"]] = 1.
    
    agg_probs = np.dot(base_probs, concord_array)
    
    #since returning aggregates
    aggregates.set_index("code", inplace=True)
    return agg_probs, aggregates