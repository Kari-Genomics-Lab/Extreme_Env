# An alignment-free analysis of genomes from extreme environments

This repository contains the representative DNA fragments and some source code used in the paper "Environment and taxonomy shape the genomic signature of prokaryotic extremophiles" available at https://www.nature.com/articles/s41598-023-42518-y

## Bacterial phylogenetic tree

![bacterial_tree](paper/Temperature_Bacteria_tree.jpg)

## Archaeal phylogenetic tree
![archaeal_tree](paper/Temperature_Archaea_tree.jpg)

## Reproducing the results

Most of the experiments in this study have been conducted using Python 3.9. To reproduce the results, make sure you have python and pip installed. It is also recommended that you create a fresh virtual environment before installing the dependencies. The use of a GPU is not required, but is is recommended. To install the dependencies, open a terminal and type in the following command.

```
pip install -r requirements.txt
```

### Data pre-processing

After the list of high-quality GTDB representative genomes are available, data pre-processing consists of the following steps:

1. Download assemblies from NCBI.

```
python DownloadAssemblies.py --assembly_file=../data/Assemblies.txt --Entrez_email=<your_email> --get_links
```

2\.1. Check missing files.\
2\.2. Remove plasmids.\
2\.3. Select one random fragment per genome.

```
python Build_Signature_Dataset.py --dataset_file=../data/Extremophiles_GTDB.tsv --fragment_len=500000 --new_folder_name=<signature_dataset_name>

```

Running the previous command should produce the same signature datasets provided in this repository.

### Supervised Learning Results

To run the 10-fold cross-validation pipeline:

```
python3 SupervisedModels.py --results_folder=../data/ --Env=Temperature --n_clusters=4
python3 SupervisedModels.py --results_folder=../data/ --Env=pH --n_clusters=2
```

To run the genera-restricted 10-fold cross-validation pipeline:

```
python3 SupervisedModels_Challenging.py --results_folder=../data/ --Env=Temperature --n_clusters=4
python3 SupervisedModels_Challenging.py --results_folder=../data/ --Env=pH --n_clusters=2
```

To run the pipeline to get the important $k$-mers:

```
python3 Important_kmers.py --folder=../data/ --dataset=pH --Env=Env --K=3
python3 Important_kmers.py --folder=../data/ --dataset=Temperature --Env=Env --K=3
```

### Unsupervised Learning Results

To run parametric clustering algorithms:

```
python3 parametric.py --results_folder=../data --Env=Temperature --n_clusters=4
python3 parametric.py --results_folder=../data --Env=pH --n_clusters=4
```

To run the non-parametric clustering algorithms (pending):


### GTDB-tk for reconstruction of the Phylogenetic trees

If you use our code or data as part of your research please cite:

```
@Article{MillanArias2023,
author={Mill{\'a}nArias, Pablo 
and Butler, Joseph
and Randhawa, Gurjit S.
and Soltysiak, Maximillian P. M.
and Hill, Kathleen A.
and Kari, Lila},
title={Environment and taxonomy shape the genomic signature of prokaryotic extremophiles},
journal={Scientific Reports},
year={2023},
month={Sep},
day={26},
volume={13},
number={1},
pages={16105},
issn={2045-2322},
doi={10.1038/s41598-023-42518-y}
}

```