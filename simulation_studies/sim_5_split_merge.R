#=========================================================================================
# SIMULATION STUDY 
#=========================================================================================
library(mcclust)
library(mcclust.ext)
library(FactoMineR)
library(factoextra)
library(gridExtra)
library(klaR)

cat('Loading functions \n');cat('\n')
source('../code/data_generation.R', echo=TRUE)
source('../code/gibbs_sampler.R', echo=TRUE)
source('../code/competitors_functions.R')
cat('Functions correctly loaded \n');cat('\n')
Sys.setenv("PKG_CXXFLAGS" = paste0('-I"C:/Users/User/AppData/Local/R/win-library/4.4/RcppGSL/include"', " -I/usr/local/include"))
# Include full library paths and libraries
Sys.setenv("PKG_LIBS" = "-L/usr/local/lib -lgsl -lgslcblas -lm")
Rcpp::sourceCpp("../code/neal8.cpp")

#=========================================================================================
# SIMULATION 5
#=========================================================================================

cat('Starting simulation set 5')

seed_set = 1

data_list_5 = list()
sim_5_gibbs = list()
psm_5_gibbs = list()
rand_VI_5_gibbs = list()
split_merge = list()
psm_5_split_merge = list()
rand_VI_5_split_merge = list()
neal8 = list()
psm_5_neal8 = list()
rand_VI_5_neal8 = list()
smn = list()
psm_5_smn = list()
rand_VI_5_smn = list()

M.na.init = 1
k.init = 2
p = 15
n = c(10,20,30)
K = 3
n_observations = sum(n)
gam = 0.3028313

for (seed in seed_set) {
  
  cat('Simulation 5.',seed,'\n')
  
  data_list_5[[seed]] = ham_mix_gen(M=c(3,4,5),
                                    k = K,
                                    p = p,
                                    n=n,
                                    s=matrix(rep(0.5,K*p),ncol = p),
                                    seed = seed)
  
  set.seed(seed)
  C.init = sample(k.init,n_observations,replace = T)
  sigma.init = matrix(runif((k.init+M.na.init)*p),nrow = (k.init+M.na.init), ncol=p)
  cent.init =  matrix(sample(k.init,(k.init+M.na.init)*p,replace = T),nrow=(k.init+M.na.init),ncol = p)
  
  
  m = apply(data_list_5[[seed]]$data,2,function(x){length(table(x))})
  u=v=vector(length = p)
  
  u[m==3]=5.00
  v[m==3]=0.25
  
  u[m==4]=4.50
  v[m==4]=0.25
  
  u[m==5]=4.25
  v[m==5]=0.25
  
  set.seed(seed)
  sim_5_gibbs[[seed]] = gibbs_mix_con(G=10000,
                                      burnin = 5000,
                                      data=data_list_5[[seed]]$data,
                                      C.init = C.init,
                                      k.init=k.init,
                                      M.na.init = M.na.init,
                                      cent.init = cent.init,
                                      sigma.init = sigma.init,
                                      u=u,
                                      v=v, 
                                      Lambda = 3,
                                      gam = gam)
  
  psm_5_gibbs[[seed]] = comp.psm(sim_5_gibbs[[seed]]$C)
  
  estim_5_VI_tmp  = mcclust.ext::minVI(psm_5_gibbs[[seed]],method = 'all',cls.draw = sim_5_gibbs[[seed]]$C)
  estim_5_VI_gibbs = estim_5_VI_tmp$cl['best',]
  rand_VI_5_gibbs[[seed]] = mcclust::arandi(estim_5_VI_gibbs,data_list_5[[seed]]$groundTruth)
  
  split_merge[[seed]] = run_markov_chain(data = data_list_5[[seed]]$data, 
                                         attrisize = m, 
                                         gamma = gam, 
                                         v = u, 
                                         w = v, 
                                         verbose = 0, 
                                         m = 3, 
                                         iterations = 10000,
                                         L = 1,
                                         c_i = C.init,
                                         burnin = 5000,
                                         t = 10, 
                                         r = 10,
                                         neal8 = FALSE,
                                         split_merge = TRUE)
  C <- matrix(unlist(lapply(split_merge[[seed]]$c_i, function(x) x + 1)), 
              nrow = 10000, 
              ncol = nrow(data_list_5[[seed]]$data), 
              byrow = TRUE)
  psm_5_split_merge[[seed]] = comp.psm(C)
  estim_5_VI_tmp  = mcclust.ext::minVI(psm_5_split_merge[[seed]],method = 'all',cls.draw = C)
  estim_5_VI_split_merge = estim_5_VI_tmp$cl['best',]
  rand_VI_5_split_merge[[seed]] = mcclust::arandi(estim_5_VI_split_merge,data_list_5[[seed]]$groundTruth)
  
  neal8[[seed]] = run_markov_chain(data = data_list_5[[seed]]$data, 
                                   attrisize = m, 
                                   gamma = gam, 
                                   v = u, 
                                   w = v, 
                                   verbose = 0, 
                                   m = 3, 
                                   iterations = 10000,
                                   L = 1,
                                   c_i = C.init,
                                   burnin = 10000,
                                   t = 10, 
                                   r = 10,
                                   neal8 = TRUE,
                                   split_merge = FALSE)
  C_n <- matrix(unlist(lapply(neal8[[seed]]$c_i, function(x) x + 1)), 
                nrow = 10000, 
                ncol = nrow(data_list_5[[seed]]$data), 
                byrow = TRUE)
  psm_5_neal8[[seed]] = comp.psm(C_n)
  estim_5_VI_tmp  = mcclust.ext::minVI(psm_5_neal8[[seed]],method = 'all',cls.draw = C_n)
  estim_5_VI_neal8 = estim_5_VI_tmp$cl['best',]
  rand_VI_5_neal8[[seed]] = mcclust::arandi(estim_5_VI_neal8,data_list_5[[seed]]$groundTruth)
  
  smn[[seed]] = run_markov_chain(data = data_list_5[[seed]]$data, 
                                 attrisize = m, 
                                 gamma = gam, 
                                 v = u, 
                                 w = v, 
                                 verbose = 0, 
                                 m = 3, 
                                 iterations = 10000,
                                 L = 1,
                                 c_i = C.init,
                                 burnin = 10000,
                                 t = 10, 
                                 r = 10,
                                 neal8 = TRUE,
                                 split_merge = TRUE)
  C_smn <- matrix(unlist(lapply(smn[[seed]]$c_i, function(x) x + 1)), 
                  nrow = 10000, 
                  ncol = nrow(data_list_5[[seed]]$data), 
                  byrow = TRUE)
  psm_5_smn[[seed]] = comp.psm(C_smn)
  estim_5_VI_tmp  = mcclust.ext::minVI(psm_5_smn[[seed]],method = 'all',cls.draw = C_smn)
  estim_5_VI_smn = estim_5_VI_tmp$cl['best',]
  rand_VI_5_smn[[seed]] = mcclust::arandi(estim_5_VI_smn,data_list_5[[seed]]$groundTruth)
  
}


#=========================================================================================
# HD_vector
#=========================================================================================
cat('Starting HD-vector \n');cat('\n')

HD=list()
HD_rand = list()
set.seed(5)
for (i in seed_set) {
  print(i)
  HD[[i]] = CategorialCluster(data_list_5[[i]]$data)
  HD_rand[[i]]=mcclust::arandi(HD[[i]][[1]],data_list_5[[i]]$groundTruth)
}

#=========================================================================================
# K-MDOES
#=========================================================================================
cat('Starting K-MODES \n');cat('\n')

K=length(unique(data_list_5[[1]]$groundTruth))
k_modes = k_modes_plusOne = k_modes_minusOne = vector('list',length = 50)
k_modes_rand =  NULL
set.seed(5)
for (i in seed_set) {
  k_modes[[i]] = kmodes(data_list_5[[i]]$data,modes = K)
  k_modes_rand[i]=mcclust::arandi(k_modes[[i]]$cluster,data_list_5[[i]]$groundTruth)
}

k_modes_plusOne_rand = NULL
set.seed(5)
for (i in seed_set) {
  k_modes_plusOne[[i]] = try(kmodes(data_list_5[[i]]$data,modes = (K+1)),TRUE)
  if(length(k_modes_plusOne[[i]])==1){
    k_modes_plusOne[[i]] = try(kmodes(data_list_5[[i]]$data,modes = (K+1)),TRUE)
  }
  if(length(k_modes_plusOne[[i]])>1){
    k_modes_plusOne_rand[i]=mcclust::arandi(k_modes_plusOne[[i]]$cluster,data_list_5[[i]]$groundTruth)
  }}

k_modes_minusOne_rand = NULL
set.seed(5)
for (i in seed_set) {
  k_modes_minusOne[[i]] = try(kmodes(data_list_5[[i]]$data,modes = (K-1)),TRUE)
  if(length(k_modes_minusOne[[i]])==1){
    k_modes_minusOne[[i]] = try(kmodes(data_list_5[[i]]$data,modes = (K-1)),TRUE)
  }
  if(length(k_modes_minusOne[[i]])>1){
    k_modes_minusOne_rand[i]=mcclust::arandi(k_modes_minusOne[[i]]$cluster,data_list_5[[i]]$groundTruth)
  }}

sim_5 = cbind(unlist(HD_rand),k_modes_minusOne_rand,k_modes_rand,k_modes_plusOne_rand,unlist(rand_VI_5_gibbs),unlist(rand_VI_5_neal8),unlist(rand_VI_5_split_merge),unlist(rand_VI_5_smn))
colnames(sim_5) = c('HD','K-Modes[2]',"K-Modes[3]","K-Modes[4]",'HMM', 'NEAL8','SPLIT&MERGE', 'SPLIT&MERGE+NEAL8')
sim_5
cat('Saving results \n');cat('\n')
save.image("output_5.RData")


