
import sys 
sys.path.append('src')
from utils import SummaryFasta, kmersFasta
import numpy as np
import argparse
import pandas as pd
from models import supervised_model
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import time

from multiprocessing import cpu_count

from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import json
import random
import torch
import os



def save(Results, dataset):
    # create json object from dictionary
    json_object = json.dumps(Results)
    # open file for writing, "w" 
    f = open(f'Supervised Results {dataset}.json',"w")
    # write json object to file
    f.write(json_object)
    # close file
    f.close()

def run(args):

    ## Read JSON results file:
    
    json_file = f'Supervised Results {args["Env"]}.json'

    if os.path.isfile(json_file) and args['continue']:
        f = open(json_file) #Supervised Results Temperature.json
  
        # returns JSON object as 
        # a dictionary
        data = json.load(f)
        Results = {}
        for key in data:
            Results[int(key)]=data[key]
        del data      
        print(Results)
    else:
        Results = {}

    ## read_fasta file
    fasta_file = f'{args["results_folder"]}/{args["Env"]}/Extremophiles_{args["Env"]}.fas'
    names, lengths, ground_truth, cluster_dis =  SummaryFasta(fasta_file, GT_file=None)
    print(len(names))


    for k in [1,2,3,4,5,6]:
        
        if not k in Results:
            Results[k] = dict()
        
        print(f'.................. k = {k} .............................')

        classifiers = (
            ("SVM", SVC),
            ("Logistic_Regression", LogisticRegression),
            ("Random Forest", RandomForestClassifier),
            #("ANN", supervised_model)
            ("ANN", MLPClassifier)
        )

        params = {
            #"ANN": {'n_clusters':args["n_clusters"], 'batch_sz':128, 'k':k, 'epochs':2},
            "ANN": {'hidden_layer_sizes':(256,64),'solver':'adam', 
                    'activation':'relu', 'alpha':1, 'learning_rate_init':0.001,
                    'max_iter':200, 'n_iter_no_change':10},  
            "SVM": {'kernel':'rbf', 'class_weight':'balanced', 'C':10},
            "Logistic_Regression": {'penalty':'l2', 'tol':0.0001, 'solver':'saga','C':20, 'class_weight':'balanced'},
            "Decisssion Tree": {},
            "Random Forest":{},
            "XG Boost": {}
        }
        
        _, kmers = kmersFasta(fasta_file, k=k, transform=None, reduce=True)
        print(f"The size of the reduced k-mer frequencies matrix is {kmers.shape}")
        kmers = np.transpose((np.transpose(kmers) / np.linalg.norm(kmers, axis=1)))
        
        n_samples, n_features = kmers.shape
        print(n_samples, n_features)

        for name, algorithm in classifiers:
            if k in Results and name in Results[k]:
                print(f"The {name} classifier was already computed for the {k}-mers")
                pass
            else:
                Results[k][name]=[0,0,0]
                
                n_splits=10
            

                print(".....Environment Cross-Validation Results .....")

                df_Env = pd.read_csv(f'{args["results_folder"]}/{args["Env"]}/Extremophiles_{args["Env"]}_GT_Env.tsv', sep='\t')
                print(len(df_Env))
                unique_labels = list(df_Env['cluster_id'].unique())
                
                
                Dataset = pd.concat([df_Env, pd.DataFrame(kmers)], axis=1)

                skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
                acc = 0

                idx = Dataset.drop_duplicates(subset=["Assembly"]).Assembly.to_numpy()
                y = Dataset.drop_duplicates(subset=["Assembly"]).cluster_id.to_numpy()
                print(len(y))

                for train, test in skf.split(idx,y):
                    model = algorithm(**params[name])
                    
                    train_Dataset = Dataset.set_index("Assembly").loc[idx[train]].reset_index()
                    test_Dataset = Dataset.set_index("Assembly").loc[idx[test]].reset_index()
                    
                    x_train = train_Dataset.loc[:,list(range(n_features))].to_numpy()
                    y_train = list(map(lambda x: unique_labels.index(x), train_Dataset.loc[:,['cluster_id']].to_numpy().reshape(-1)))
                    
                    
                    x_test = test_Dataset.loc[:,list(range(n_features))].to_numpy()
                    y_test = list(map(lambda x: unique_labels.index(x), test_Dataset.loc[:,['cluster_id']].to_numpy().reshape(-1)))
                    
                    model.fit(x_train, y_train)
                    
                    score = model.score(x_test , y_test)
                    #print(score)
                    acc += score
                    del model
                    
                Results[k][name][0] = acc/n_splits 
                print(name, acc/n_splits)
                save(Results, args["Env"])

                
                
                print("..... Random Labelling .....")

                skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
                
                acc = 0
                
                #Dataset['random_label'] = Dataset.assign(random_label=0).groupby(Dataset['Assembly'])['random_label'].transform(lambda x: np.random.randint(0, args['n_clusters'], len(x)))
                #print(Dataset)
                Dataset['random_label'] = Dataset['cluster_id']
                Dataset['random_label'] = Dataset['random_label'].sample(frac=1, ignore_index=True)

                idx = Dataset.drop_duplicates(subset=["Assembly"]).Assembly.to_numpy()
                y = Dataset.drop_duplicates(subset=["Assembly"]).random_label.to_numpy()
                #print(y)
                #random = np.random.choice([0,1,2,3], y.shape[0])
                

                for train, test in skf.split(idx,y):
                    model = algorithm(**params[name])
                    train_Dataset = Dataset.set_index("Assembly").loc[idx[train]].reset_index()
                    test_Dataset = Dataset.set_index("Assembly").loc[idx[test]].reset_index()
                    
                    x_train = train_Dataset.loc[:,list(range(n_features))].to_numpy()
                    y_train = list(map(lambda x: unique_labels.index(x), train_Dataset.loc[:,['random_label']].to_numpy().reshape(-1)))
                    #print(y_train)
                    
                    x_test = test_Dataset.loc[:,list(range(n_features))].to_numpy()
                    y_test = list(map(lambda x: unique_labels.index(x), test_Dataset.loc[:,['random_label']].to_numpy().reshape(-1)))
                    #print(y_test)

                    
                    model.fit(x_train, y_train)
                    
                    score = model.score(x_test , y_test)
                    #print(score)
                    acc += score
                    del model
                Results[k][name][2] = acc/n_splits
                save(Results, args["Env"])
                print(name, acc/n_splits)

                print(".... Taxonomy Cross Validation Results .....")

                df_Tax = pd.read_csv(f'{args["results_folder"]}/{args["Env"]}/Extremophiles_{args["Env"]}_GT_Tax.tsv', sep='\t')
                unique_labels = list(df_Tax['cluster_id'].unique())
                Dataset = pd.concat([df_Tax, pd.DataFrame(kmers)], axis=1)
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
                acc = 0

                idx = Dataset.drop_duplicates(subset=["Assembly"]).Assembly.to_numpy()
                y = Dataset.drop_duplicates(subset=["Assembly"]).cluster_id.to_numpy()

                for train, test in skf.split(idx,y):
                    model = algorithm(**params[name])
                    train_Dataset = Dataset.set_index("Assembly").loc[idx[train]].reset_index()
                    test_Dataset = Dataset.set_index("Assembly").loc[idx[test]].reset_index()
                    
                    x_train = train_Dataset.loc[:,list(range(n_features))].to_numpy()
                    y_train = list(map(lambda x: unique_labels.index(x), train_Dataset.loc[:,['cluster_id']].to_numpy().reshape(-1)))
                    
                    
                    x_test = test_Dataset.loc[:,list(range(n_features))].to_numpy()
                    y_test = list(map(lambda x: unique_labels.index(x), test_Dataset.loc[:,['cluster_id']].to_numpy().reshape(-1)))
                    
                    model.fit(x_train, y_train)
                    
                    score = model.score(x_test , y_test)
                    #print(score)
                    acc += score
                    del model
                    
                Results[k][name][1] = acc/n_splits 
                print(name, acc/n_splits)
                save(Results, args["Env"])
                #result = permutation_importance(forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
                #rint(result.importances_mean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder', action='store', type=str)
    parser.add_argument('--Env', action='store', type=str) #[Temperature, pH]
    parser.add_argument('--n_clusters',action='store', type=int, default=None) #[int]
    parser.add_argument('--continue',action='store_true')
    args = vars(parser.parse_args())

    torch.set_num_threads(cpu_count() - 2 )   
    run(args)


if __name__ == '__main__':
    main()