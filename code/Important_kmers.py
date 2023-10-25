#python Important_kmers.py --folder=../Refactor_Single_Fragment/500k --dataset=pH --Env=Env --K=3
# python Important_kmers.py --folder=../data/pH --dataset=pH --Env=Env --K=3

import sys 
sys.path.append('src')

import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors 


from sklearn.model_selection import  StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier


from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# import plot_functions as plots
import matplotlib.pyplot as plt
from itertools import product

from utils import SummaryFasta, kmersFasta
import pandas as pd
import random
import argparse

import matplotlib

#matplotlib.use('PS')

#pH
np.random.seed(321)
random.seed(321)


plt.rcParams.update({
   "text.usetex": True,
   "font.family": "serif",
   "font.serif": ["Times"],
})

#matplotlib.rcParams['figure.dpi'] = 300




codontab = {
    'TCA': 'S',    # Serina
    'TCC': 'S',    # Serina
    'TCG': 'S',    # Serina
    'TCT': 'S',    # Serina
    'TTC': 'F',    # Fenilalanina
    'TTT': 'F',    # Fenilalanina
    'TTA': 'L',    # Leucina
    'TTG': 'L',    # Leucina
    'TAC': 'Y',    # Tirosina
    'TAT': 'Y',    # Tirosina
    'TAA': '*',    # Stop
    'TAG': '*',    # Stop
    'TGC': 'C',    # Cisteina
    'TGT': 'C',    # Cisteina
    'TGA': '*',    # Stop
    'TGG': 'W',    # Triptofano
    'CTA': 'L',    # Leucina
    'CTC': 'L',    # Leucina
    'CTG': 'L',    # Leucina
    'CTT': 'L',    # Leucina
    'CCA': 'P',    # Prolina
    'CCC': 'P',    # Prolina
    'CCG': 'P',    # Prolina
    'CCT': 'P',    # Prolina
    'CAC': 'H',    # Histidina
    'CAT': 'H',    # Histidina
    'CAA': 'Q',    # Glutamina
    'CAG': 'Q',    # Glutamina
    'CGA': 'R',    # Arginina
    'CGC': 'R',    # Arginina
    'CGG': 'R',    # Arginina
    'CGT': 'R',    # Arginina
    'ATA': 'I',    # Isoleucina
    'ATC': 'I',    # Isoleucina
    'ATT': 'I',    # Isoleucina
    'ATG': 'M',    # Methionina
    'ACA': 'T',    # Treonina
    'ACC': 'T',    # Treonina
    'ACG': 'T',    # Treonina
    'ACT': 'T',    # Treonina
    'AAC': 'N',    # Asparagina
    'AAT': 'N',    # Asparagina
    'AAA': 'K',    # Lisina
    'AAG': 'K',    # Lisina
    'AGC': 'S',    # Serina
    'AGT': 'S',    # Serina
    'AGA': 'R',    # Arginina
    'AGG': 'R',    # Arginina
    'GTA': 'V',    # Valina
    'GTC': 'V',    # Valina
    'GTG': 'V',    # Valina
    'GTT': 'V',    # Valina
    'GCA': 'A',    # Alanina
    'GCC': 'A',    # Alanina
    'GCG': 'A',    # Alanina
    'GCT': 'A',    # Alanina
    'GAC': 'D',    # Acido Aspartico
    'GAT': 'D',    # Acido Aspartico
    'GAA': 'E',    # Acido Glutamico
    'GAG': 'E',    # Acido Glutamico
    'GGA': 'G',    # Glicina
    'GGC': 'G',    # Glicina
    'GGG': 'G',    # Glicina
    'GGT': 'G'     # Glicina
}


def reverse_complement(x, k):
    numbits = 2*k  
    mask = 0xAAAAAAAA
    x = ((x >> 1) & (mask>>1)) | ((x<< 1) & mask)
    x = (1 << numbits) - 1 - x
    rev = 0

    size = 2**numbits-1
    while(size > 0):
        rev <<= 1
        if x & 1 == 1:
            rev ^= 1
        x >>=1
        size >>= 1

    return rev

def kmer_rev_comp(kmer_counts, k):
    index=[]
    for kmer in range(4**k):
        revcomp = reverse_complement(kmer,k)

        # Only look at canonical kmers - this makes no difference
        if kmer <= revcomp:
            index.append(kmer)
            kmer_counts[kmer] += kmer_counts[revcomp]
            kmer_counts[kmer] *= 0.5
        #else:
        #    kmer_counts[kmer] = kmer_counts[revcomp]

    return kmer_counts[index]



def kmer2idx(kmer, reduce=False):
    encoding = {'A':0, 'C':1, 'G':2, 'T':3}
    size = (1 << (2 * len(kmer))) - 1  

    idx = 0 
    for i in range(len(kmer)):
        bp = kmer[i]
        bp_code = encoding[bp]
        idx = ((idx << 2) | bp_code) & size

    return idx

def pos_gen(kmer):
    """
    Find the position of a particular kmer in the CGR.
    :param kmer: string with the kmer.
    :return: position in the CGR.
    """
    k = len(kmer)

    posx = 2 ** k
    posy = 2 ** k

    for i in range(1, k + 1):
        bp = kmer[-i]
        if bp == 'C':
            posx = posx - 2 ** (k - i)
            posy = posy - 2 ** (k - i)

        elif bp == 'A':
            posx = posx - 2 ** (k - i)

        elif bp == 'G':
            posy = posy - 2 ** (k - i)

    return int(posx - 1), int(posy - 1)


def cgr_gen(probs, k):
    """
    Generate CGR from the kmer counts for a given value of k.
    :param probs: array with the normalized kmer counts
    :param k:
    :return: 2D - CGR pattern.
    """
    kamers = product('ACGT', repeat=k)
    mat = np.zeros((2 ** k, 2 ** k))

    for i, kmer in enumerate(kamers):
        x, y = pos_gen(kmer)
        mat[y][x] = probs[i]

    return mat



def run(args):
    summary_file = '../data/Extremophiles_GTDB.tsv'
    summary_dataset = pd.read_csv(summary_file, sep='\t')
    summary_dataset = pd.read_csv(summary_file, sep='\t', usecols=["Domain", "Temperature", "Assembly", "pH", "Genus", "Species"])
    print(summary_dataset.groupby(['Domain', 'pH']).nunique())
    print(summary_dataset.groupby(['Domain', 'Temperature']).nunique())

    ## read_fasta file
    fasta_file = f'{args["folder"]}/{args["dataset"]}/Extremophiles_{args["dataset"]}.fas'

    names, lengths, ground_truth, cluster_dis =  SummaryFasta(fasta_file, GT_file=None)

    dataset = args["dataset"]


    df = pd.read_csv(f'{args["folder"]}/{args["dataset"]}/Extremophiles_{dataset}_GT_Env.tsv', sep='\t')
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

    k = args['K']

    ## Preparing the k-mers

    all_kmers = product('ACGT', repeat=k)
    all_kmers = np.array([''.join(x) for x in all_kmers])


    canonical=[]
    for i, kmer in enumerate(all_kmers):
        revcomp = reverse_complement(i,k)

        # Find
        if i <= revcomp:
            canonical.append(kmer)
        
    #print(canonical)
    canonical = np.array(canonical)

    params = {
            #"ANN": {'n_clusters':args["n_clusters"], 'batch_sz':512, 'k':k, 'epochs':100}, 
            "ANN": {'hidden_layer_sizes':(256,64), 'solver':'adam', 
                    'activation':'relu', 'alpha':1, 'learning_rate_init':0.001, 'max_iter':1000},
            "SVM": {'kernel':'rbf', 'class_weight':'balanced', 'C':10},
            "Logistic_Regression": {'penalty':'l2', 'tol':0.0001, 'solver':'saga','C':20, 'class_weight':'balanced'},
            "Decisssion Tree": {},
            "Random Forest":{},
            "XG Boost": {}
        }

    
    print(f'.................. k = {k} .............................')
    
    _, kmers = kmersFasta(fasta_file, k=k, transform=None, reduce=True)
    _mean = np.mean(kmers, axis=0)
    #kmers = np.transpose((np.transpose(kmers) / np.linalg.norm(kmers, axis=1)))
    
    n_samples, n_features = kmers.shape
    n_splits = 10
    print(n_samples, n_features)

    Differential=[]

    name="Random Forest"
    algorithm=RandomForestClassifier
            
    print(".....Environment Cross-Validation Results .....")

    df_Env = pd.read_csv(f'{args["folder"]}/{args["dataset"]}/Extremophiles_{args["dataset"]}_GT_{args["Env"]}.tsv', sep='\t')
    
    if args["Env"] =="Env":
        if args["dataset"] == 'pH':
            unique_labels =  sorted(list(df_Env['cluster_id'].unique()), reverse=True) 
        else:
            unique_labels = ['Psychrophiles','Mesophiles', 'Thermophiles', 'Hyperthermophiles']
        numClasses = len(unique_labels)
    else:
        unique_labels=["Bacteria", "Archaea"]

    print(unique_labels)

    Dataset = pd.concat([df_Env, pd.DataFrame(kmers)], axis=1)

    
    for label in unique_labels:
        print(label)

        importance=[]
        skf =  StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
        acc = 0
    
        GT = Dataset.cluster_id.to_numpy()
        y = (GT == label)


        for train, test in skf.split(_,y, groups=data["genus"].values):
        
            model = algorithm(**params[name])

            train_Dataset = Dataset.iloc[train]
            test_Dataset = Dataset.iloc[test]

            x_train = train_Dataset.loc[:,list(range(n_features))].to_numpy()
            y_train = y[train]
            
            
            x_test = test_Dataset.loc[:,list(range(n_features))].to_numpy()
            y_test = y[test]
            
            model.fit(x_train, y_train)
            importance.append(model.feature_importances_)
            
            score = model.score(x_test , y_test)
            #print(score)
            acc += score
            del model
        
        acc /= n_splits 
        print(name,label, acc)
    
        importance = np.array(importance)
        #print(importance)
        importance_mean = np.mean(importance, axis=0)
        #print(importance_mean)
        ind = importance_mean.argsort()[::-1]


        if k == 3: 
            new_score = 0
            i =  1
            new_acc = 0
            while (new_acc < acc - 0.01) and i < kmers.shape[1]:
                skf =  StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
                new_acc = 0
                for train, test in skf.split(_,y, groups=data["genus"].values):

                    model = algorithm(**params[name])

                    train_Dataset = Dataset.iloc[train]
                    test_Dataset = Dataset.iloc[test]

                    x_train = train_Dataset.loc[:,ind[:i]].to_numpy()
                    y_train = y[train]
                    
                    
                    x_test = test_Dataset.loc[:,ind[:i]].to_numpy()
                    y_test = y[test]
                    
                    model.fit(x_train, y_train)
                
                    new_score = model.score(x_test , y_test)
                    new_acc += new_score
                    del model
                new_acc /= n_splits
                print(new_acc)
                
                i += 1

            print("Truly Important:", i)
            print(canonical[ind[:i]])
            Differential.append(ind[:i])

        else:
            Differential.append(importance_mean)
        

    if k == 3:
        plt.figure(figsize=(10,10))    #(10,5)
        plot_num = 1
        for i, relevant in enumerate(Differential):
            color_bar = np.array(['royalblue']* kmers.shape[1])
            heights = (kmers[GT==unique_labels[i] ,:].mean(axis=0)-_mean)

            color_bar[relevant]='palegreen'
            plt.subplot(len(unique_labels),1, plot_num)
            plt.bar(canonical,100*heights,color=color_bar)
            plt.tick_params(axis='x', labelsize=10, labelrotation=45)
            #plt.ylim(-0.015, 0.015)
            plt.yticks(np.arange(-1.5, 1.5, step=0.25))  # Set label locations.
            plt.ylabel('Deviation (\%)')
            #plt.yticks(np.arange(-0.015, 0.017, step=0.005)) 
            plt.title(unique_labels[i])
            #if plot_num == 1:
            legend_elements = [Patch(facecolor='palegreen',label='Relevant'),
                                Patch(facecolor='royalblue', label='Not Relevant')]
            plt.legend(handles=legend_elements)
            plot_num += 1
        

        #plt.suptitle(f'Relevant $k$-mers to separate by taxonomy \n and deviation from the mean histogram in {dataset} Dataset ($k=$ {k})', fontsize=16, fontweight="heavy", multialignment='center')
        #plt.suptitle(f'Distinction of Relevant $3$-mers for classification in the {dataset} Dataset. \n Deviation from the Mean Histogram expresed in Percentage Points.', fontsize=16, fontweight="heavy", multialignment='center')
        
        #plt.suptitle(f'Deviation from the mean histogram of $3$-mers for each environment category in the {dataset} dataset', fontsize=16, fontweight="heavy", multialignment='center')
        plt.suptitle(f'Histograms illustrating the deviation of the $3$-mer counts in each environment category \n from the mean $3$-mer counts in the {dataset} Dataset', fontsize=16, fontweight="heavy", multialignment='center')
        
        # Create the figure
        
        plt.tight_layout()
        
        #plt.savefig(f'Histograms_{k}_{dataset}_Env.jpeg', dpi=300)
        #plt.savefig(f'Histograms_{k}_{dataset}_Env.svg', dpi=300, format='svg')
    
        df = pd.DataFrame(dict([(unique_labels[i],pd.Series(canonical[v])) for i,v in enumerate(Differential)]))
        #df.to_csv(f'{args["folder"]}/k_{k}_{dataset}_{args["Env"]}.tsv', sep="\t", index=False)

    else:
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(3*4, 4.5)) #3*numclases, 4.5
        plot_num = 1

        for i in range(4):
            ax[i].set_axis_off()
        
        if dataset == 'pH':
            offset = 1
        else: 
            offset = 0
        
        for i, relevant in enumerate(Differential):    
            importance_cgr = np.zeros(4**k)
            for j, kmer in enumerate(canonical):
                true_idx = kmer2idx(kmer)
                importance_cgr[true_idx] = relevant[j]
                importance_cgr[reverse_complement(true_idx, k)] = relevant[j]
  
            ax[i+offset].set_title(f'{unique_labels[i]}')
            ax[i+offset].set_axis_off()
            ax[i+offset].text(-6,-0.1,"C",fontsize=12)
            ax[i+offset].text(64,-0.1,"G",fontsize=12)
            ax[i+offset].text(-6,64.4,"A",fontsize=12)
            ax[i+offset].text(64,64.4,"T",fontsize=12)
            

            connectivity = cgr_gen(importance_cgr, k)
            connectivity = (connectivity-np.min(connectivity))/(np.max(connectivity) - np.min(connectivity))

            im = ax[i+offset].imshow(connectivity, cmap = 'GnBu') #GnBu
            plot_num += 1

            # fig_2, ax_2 = plt.subplots(nrows=1, ncols=1)
            # ax_2.imshow(connectivity, cmap = 'GnBu') #GnBu
            # ax_2.set_title(f'{unique_labels[i]}')
            # ax_2.set_axis_off()
            # ax_2.text(-6,-0.1,"C",fontsize=12)
            # ax_2.text(64,-0.1,"G",fontsize=12)
            # ax_2.text(-6,64.4,"A",fontsize=12)
            # ax_2.text(64,64.4,"T",fontsize=12)
            # plt.show()

        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.20, orientation='horizontal', fraction=0.1, location='bottom', aspect=25)
        #plt.suptitle(f'Frequency Chaos Game Representation ($fCGR_{k}$) of the global importance \n of different 6-mers distinguishing each environment category in the {dataset} Dataset \n', fontsize=14, fontweight='black')
        #plt.suptitle(f'$fCGR_{k}$ illustrating the global importance of each 6-mer in the classification \n of DNA sequences of each environment category from the rest in the {dataset} Dataset', fontsize=14, fontweight='black')
        #plt.savefig(f'Signatures_{k}_{dataset}_{args["Env"]}.png', dpi=150)
        #plt.savefig(f'Signatures_{k}_{dataset}_{args["Env"]}.svg', dpi=300)
    #fig.set_tight_layout(True)
    plt.tight_layout()
    plt.show()
    

def main():
    
    parser= argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', action='store', type=str)
    parser.add_argument('--dataset', action='store', type=str) #Temperature or pH
    parser.add_argument('--Env', action ='store', type=str) #Tax or Env
    parser.add_argument('--K', action ='store', type=int)

    args = vars(parser.parse_args())
    print(':)')
    run(args)

if __name__ == '__main__':
    main()




