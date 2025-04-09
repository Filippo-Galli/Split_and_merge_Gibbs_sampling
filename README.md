# Split-and-Merge sampling algorithm for Hamming-mixture models of categorical data

## 🔍 Introduction
This repository contains the code to reproduce the result of the pubblication "Split-and-Merge sampling algorithm for
Hamming-mixture models of categorical data" made by Di Marino et al. (2025) on the SIS conference of Genova.
> **ABSTRACT:** \
> This work aims to design a Gibbs sampling algorithm for posterior Bayesian inference of a Dirichlet process mixture model based on Hamming distributed kernels, a probability measure built upon the Hamming distance. This model is employed to provide model-based clustering analysis of categorical data with no natural ordering. The proposed algorithm leverages a split-and-merge Markov chain Monte Carlo technique to address the curse of dimensionality issue and improve the search over the space of random partitions. 

## 📂 Repository Structure 
```
Split_and_merge_Gibbs_sampling /
│
├── code/                       # Folder with all the code used
│   └── old_code/               # Additional materials
│
├── realdata_analysis/          # Folder with the entry point for the analysis
│   ├── digits_simulator.R      # R script to perform analysis on the MNIST Dataset
|   └── zoo_simulator.R         # R script to perform analysis on the Zoo Dataset
│
└── data/                       # Folder with the dataset tested
```

##  ▶️ How to run the simulation 
Before running the simulation you need to create the folder `print/` and `results/` in the root directory of the project. In the first one will be saved the figures created by the posterior analysis of the chain while in the second one will be saved the RData file with some information saved during program execution.  \
To run the simulation of zoo dataset, you need to run the `zoo_simulator.R` script. Inside of it you can modify all the parameters in the appropriate section. 

## 🏗️ Code organization
As you can see from the structure of the repository, the code is organized in two main folders: `code/` and `realdata_analysis/`. The first one contains all the code used to implement the Split-and-Merge algorithm, while the second one contains the entry point for the analysis. \
The `realdata_analysis/` folder contains the entry point for the analysis. Inside of it you can find the `digits_simulator.R` and `zoo_simulator.R` scripts, which are used to run the simulation on the MNIST and Zoo datasets, respectively. 

##  🤝 To contribute
To modify the code we have a unique file for the implementation of the Split-and-Merge algorithm, which is `split_and_merge.cpp` with its header. This file contains all the functions used to implement the algorithm. To see the code related to Algorithm 8 of Neal (2000) there is the file `neal8.cpp` with its respective header. We used the file `common_functions.cpp` and its header to avoid code repetitions between the two algorithms. The remaining file is `hyperg.cpp` and its header which contains the implementation of the hypergeometric-inverse-gamma distribution and its approximation with the beta distribution.

## ✨ Interesting features not present in the paper
For the sake of linearity, we decided to not include some interesting features in the paper. Here are some of them:
- It include a rustic ReUse Algorithm to improve perfomance of Algorithm 8 of Neal (2000). 
- It contain in `hyperg.cpp` an approximation of the hypergeometric-inverse-gamma distribution with the beta distribution used dynamically with certain conditions are satisfied.

## 👥 Acknowledgements
The first part of this project was developed for the course of "Bayesian Statistics" of Politecnico of Milano. A special thanks to the starting team composed by: Sara di Marino which help in the development of the code, Claudio Barbieri to link the theoretical part with the code, the group of theorists Roberto Fortuna, Benedetta Sabina Leone and Leonardo Grifalconi. \
A big thanks also to our supervisor prof. Argiento Raffaele, dott. Cremaschi Andrea and prof. Paci Lucia for their support, help and supervision during the development of this project.

## 🏁 Future improvements
We also start with the code refactoring to exploit the use of the OOP paradigm to improve the code readibility, modularity and reusability.

##  📖 References
- Argiento, R.,  De Iorio M. Is infinity that far? A Bayesian nonparametric perspective of finite mixture models. Ann. Statist. 50, 2641-2663 (2022).
- Argiento, R., Filippi-Mazzola, E., Paci L. Model-based clustering of
categorical data based on the Hamming distance. J. Amer. Statist. Assoc.  doi:10.1080/01621459.2024.2402568 (2024)
- Ghilotti, L., Beraha M., Guglielmi A.  Bayesian clustering of high-dimensional data via latent repulsive mixtures. Biometrika, doi: 10.1093/biomet/asae059 (2024).
- Hamming, R. W. Error detecting and error correcting codes. Bell Syst. Tech. J. 29, 147-160 (1950).
- Hubert, L., Arabie P. Comparing partitions. J. Classif. 2, 193-218 (1985).
- Jain, S., Neal R. M. Splitting and merging components of a nonconjugate Dirichlet process mixture model. Bayesian Anal. 2, 445-472 (2007).
- Neal, R. M. Markov chain sampling methods for Dirichlet process mixture models. J. Comput. Graph. Stat. 9, 249-265 (2000).
- Stefano Favaro, Yee Whye Teh "MCMC for Normalized Random Measure Mixture Models," Statistical Science, Statist. Sci. 28(3), 335-359, (August 2013)
