import sys 
sys.path.append('src')
from utils import SummaryFasta, kmersFasta
import numpy as np
import argparse
import pandas as pd
from models import supervised_model
from sklearn.model_selection import KFold, StratifiedGroupKFold, train_test_split
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
import json
import random
import torch
import os



def save(Results, dataset):
    # create json object from dictionary
    json_object = json.dumps(Results)
    # open file for writing, "w" 
    f = open(f'Supervised Results no genus {dataset}.json',"w")
    # write json object to file
    f.write(json_object)
    # close file
    f.close()

def run(args):

    ## Read JSON results file:
    
    json_file = f'Supervised Results no Genus {args["Env"]}.json'

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

    summary_file = f"{args['results_folder']}/Extremophiles_GTDB.tsv"
    summary_dataset = pd.read_csv(summary_file, sep='\t')
    summary_dataset = pd.read_csv(summary_file, sep='\t', usecols=["Domain", "Temperature", "Assembly", "pH", "Genus", "Species"])
    print(summary_dataset.groupby(['Domain', 'pH']).nunique())
    print(summary_dataset.groupby(['Domain', 'Temperature']).nunique())
    
     
    ## read_fasta file
    fasta_file = f'{args["results_folder"]}/{args["Env"]}/Extremophiles_{args["Env"]}.fas'
    names, lengths, ground_truth, cluster_dis =  SummaryFasta(fasta_file, GT_file=None)

    dataset = args["Env"]

    df = pd.read_csv(f'{args["results_folder"]}/{args["Env"]}/Extremophiles_{dataset}_GT_Env.tsv', sep='\t')
    df.rename(columns = {'cluster_id':dataset}, inplace = True)

    assemblies = []
    genera = []
    species = []
    domain = []
    env = []

    assembly_dict = dict(zip(df['sequence_id'], df['Assembly']))

    for name in names:
        row = summary_dataset.loc[summary_dataset["Assembly"] == assembly_dict[name]]
        #print(name)
        assemblies.append(row["Assembly"].values[0])
        genera.append(row["Genus"].values[0])
        species.append(row["Species"].values[0])
        domain.append(row["Domain"].values[0])
        env.append(row[dataset].values[0])
        
    data = pd.DataFrame.from_dict({'Domain':domain, dataset:env, "sequence_id":names,
                                   "Assembly":assemblies, "genus":genera, "species":species})
    
    
    n_splits=10    

    for k in [1,2,3,4,5,6]:
        Results[k] = dict()
        
        print(f'.................. k = {k} .............................')

        classifiers = (
            ("SVM", SVC),
            ("Logistic_Regression", LogisticRegression),
            ("Random Forest", RandomForestClassifier),
            ("ANN", supervised_model)
        )

        params = {
            "ANN": {'n_clusters':args["n_clusters"], 'batch_sz':128, 'k':k, 'epochs':100}, 
            "SVM": {'kernel':'rbf', 'class_weight':'balanced', 'C':10},
            "Logistic_Regression": {'penalty':'l2', 'tol':0.0001, 'solver':'saga','C':20, 'class_weight':'balanced'},
            "Decisssion Tree": {},
            "Random Forest":{},
            "XG Boost": {}
        }
        
        _, kmers = kmersFasta(fasta_file, k=k, transform=None, reduce=True)
        #kmers = np.transpose((np.transpose(kmers) / np.linalg.norm(kmers, axis=1)))
        
        n_samples, n_features = kmers.shape
        print(n_samples, n_features)

        for name, algorithm in classifiers:
            
            Results[k][name]=[0,0,0]
            
            
            print(".....Environment Cross-Validation Results .....")

            df_Env = pd.read_csv(f'{args["results_folder"]}/{args["Env"]}/Extremophiles_{args["Env"]}_GT_Env.tsv', sep='\t')
            
            unique_labels = list(df_Env['cluster_id'].unique())
            
            
            Dataset = pd.concat([df_Env, pd.DataFrame(kmers)], axis=1)

            skf =  StratifiedGroupKFold(n_splits=n_splits)
            acc = 0

            #idx = Dataset.drop_duplicates(subset=["Assembly"]).Assembly.to_numpy()
            y = Dataset.cluster_id.to_numpy()
            #print(y)

            for train, test in skf.split(_,y, groups=data["genus"].values):
                
                model = algorithm(**params[name])
 
                train_Dataset = Dataset.iloc[train]
                test_Dataset = Dataset.iloc[test]

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

            skf = StratifiedGroupKFold(n_splits=n_splits)
            
            acc = 0
            
            #Dataset['random_label'] = Dataset.assign(random_label=0).groupby(Dataset['Assembly'])['random_label'].transform(lambda x: np.random.randint(0, args['n_clusters'], len(x)))
            #print(Dataset)
            
            Dataset['random_label'] = Dataset['cluster_id']
            Dataset['random_label'] = Dataset['random_label'].sample(frac=1, ignore_index=True)

            #idx = Dataset.drop_duplicates(subset=["Assembly"]).Assembly.to_numpy()
            y = Dataset.random_label.to_numpy()
            #print(y)
            #random = np.random.choice([0,1,2,3], y.shape[0])
            

            for train, test in skf.split(_,y, groups=data["genus"].values):
                model = algorithm(**params[name])

                train_Dataset = Dataset.iloc[train]
                test_Dataset = Dataset.iloc[test]
                
                x_train = train_Dataset.loc[:,list(range(n_features))].to_numpy()
                y_train = list(map(lambda x: unique_labels.index(x), train_Dataset.loc[:,['random_label']].to_numpy().reshape(-1)))
                
                
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
            skf = StratifiedGroupKFold(n_splits=n_splits)
            acc = 0

            y = Dataset.cluster_id.to_numpy()

            for train, test in skf.split(_,y, groups=data["genus"].values):
                model = algorithm(**params[name])
                
                train_Dataset = Dataset.iloc[train]
                test_Dataset = Dataset.iloc[test]

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