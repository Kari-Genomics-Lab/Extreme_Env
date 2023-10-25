
import sys
import os
import numpy as np
import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


sys.path.append('src')
from utils import SummaryFasta, cluster_acc, kmersFasta, modified_cluster_acc
from idelucs.cluster import iDeLUCS_cluster


import argparse


def build_pipeline(numClasses, method):
    normalizers = []
    if method == 'GMM':
        normalizers = [('classifier', mixture.GaussianMixture(n_components=numClasses))]
    if method == 'k-means++':
        normalizers.append(('classifier', KMeans(n_clusters=numClasses, init='k-means++', random_state=321)))
    if method == 'k-medoids':
        normalizers.append(('classifier', KMedoids(n_clusters=numClasses)))
    return Pipeline(normalizers)


def K_MEANS(sequence_file, n_cluster, k=6):
    sys.stdout.write("......... k-means...............\n")
    sys.stdout.flush()
    _, kmers = kmersFasta(sequence_file, k=k)
    pipeline = build_pipeline(n_cluster, 'k-means++')
    y_pred = pipeline.fit_predict(kmers)
   
    return y_pred

def K_MEDOIDS(sequence_file, n_cluster, k=6):
    sys.stdout.write("......... k-medoids...............\n")
    sys.stdout.flush()
    _, kmers = kmersFasta(sequence_file, k=k)
    pipeline = build_pipeline(n_cluster, 'k-medoids')
    y_pred = pipeline.fit_predict(kmers)
   
    return y_pred  

def GMM(sequence_file, n_cluster, k=6):
    sys.stdout.write("......... Gaussian Mixture ...............\n")
    sys.stdout.flush()
    names, kmers = kmersFasta(sequence_file, k=6)
    pipeline = build_pipeline(n_cluster, 'GMM')
    y_pred = pipeline.fit_predict(kmers)
    return y_pred

def iDeLUCS(sequence_file,n_clusters, k=6):
    print("......... iDeLUCS ...............\n")
    sys.stdout.flush()
    params = {'sequence_file':sequence_file,'n_clusters':n_clusters, 'n_epochs':50,
            'n_mimics':3, 'batch_sz':512, 'k':k, 'weight':0.5, 'n_voters':5}
            
    model = iDeLUCS_cluster(**params)
    labels, latent = model.fit_predict(sequence_file)
    return labels

def run(args):

    dataset = args["Env"]

    sequence_file = f'{args["results_folder"]}/{args["Env"]}/Extremophiles_{args["Env"]}.fas' 
    GT_file = f'{args["results_folder"]}/{args["Env"]}/Extremophiles_{dataset}_GT_Env.tsv'

    names, lengths, GT, cluster_dis = SummaryFasta(sequence_file, GT_file)

    unique_labels = list(np.unique(GT))
    numClasses = len(unique_labels)
    y_true = np.array(list(map(lambda x: unique_labels.index(x), GT)))


    clust_algorithms = [("k-means", K_MEANS), 
                        ("k-medoids", K_MEDOIDS),
                        ("GMM", GMM),
                        ("iDeLUCS", iDeLUCS)]

    for name, clust_algo in clust_algorithms:
        assignments = clust_algo(sequence_file, args["n_clusters"])
        ind, acc = cluster_acc(assignments, y_true)
        print(f"{name}: {acc*100}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clusters', action='store',type=int, default=4)
    parser.add_argument('--results_folder', action='store', type=str)
    parser.add_argument('--Env', action='store',type=str,default=None)

    args = vars(parser.parse_args())
    run(args)


if __name__ == '__main__':
    main()
