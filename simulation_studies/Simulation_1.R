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

#=========================================================================================
# SIMULATION 1
#=========================================================================================

cat('Starting simulation set 1')

seed_set = 1:50

data_list_1 = list()
sim_1       = list()
psm_1       = list()
rand_VI_1   = list()


M.na.init = 1
k.init = 2
p = 15
nj = 25
K = 3

for (seed in seed_set) {
  
  cat('Simulation 1.',seed,'\n')
  
  data_list_1[[seed]] = ham_mix_gen(M=c(3,4,5),
                                    k = K,
                                    p = p,
                                    n=c(nj,nj,nj),
                                    s=matrix(rep(0.2,K*p),ncol = p),
                                    seed = seed)
  
  set.seed(seed)
  C.init = sample(k.init,nj*K,replace = T)
  sigma.init = matrix(runif((k.init+M.na.init)*p),nrow = (k.init+M.na.init), ncol=p)
  cent.init =  matrix(sample(k.init,(k.init+M.na.init)*p,replace = T),nrow=(k.init+M.na.init),ncol = p)
  
  
  m = apply(data_list_1[[seed]]$data,2,function(x){length(table(x))})
  u=v=vector(length = 15)
  
  u[m==3]=5.00
  v[m==3]=0.25
  
  u[m==4]=4.50
  v[m==4]=0.25
  
  u[m==5]=4.25
  v[m==5]=0.25
  
  set.seed(seed)
  sim_1[[seed]] = gibbs_mix_con(G=10000,
                                burnin = 5000,
                                data=data_list_1[[seed]]$data,
                                C.init = C.init,
                                k.init=k.init,
                                M.na.init = M.na.init,
                                cent.init = cent.init,
                                sigma.init = sigma.init,
                                u=u,
                                v=v, 
                                Lambda = 3,
                                gam = 0.1905278)
  
  psm_1[[seed]] = comp.psm(sim_1[[seed]]$C)
  
  estim_1_VI_tmp  = mcclust.ext::minVI(psm_1[[seed]],method = 'all',cls.draw = sim_1[[seed]]$C)
  estim_1_VI = estim_1_VI_tmp$cl['best',]
  rand_VI_1[[seed]] = mcclust::arandi(estim_1_VI,data_list_1[[seed]]$groundTruth)
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
  HD[[i]] = CategorialCluster(data_list_1[[i]]$data)
  HD_rand[[i]]=mcclust::arandi(HD[[i]][[1]],data_list_1[[i]]$groundTruth)
}

#=========================================================================================
# K-MDOES
#=========================================================================================
cat('Starting K-MODES \n');cat('\n')

K=length(unique(data_list_1[[1]]$groundTruth))
k_modes = k_modes_plusOne = k_modes_minusOne = vector('list',length = 50)
k_modes_rand =  NULL
set.seed(5)
for (i in seed_set) {
  k_modes[[i]] = kmodes(data_list_1[[i]]$data,modes = K)
  k_modes_rand[i]=mcclust::arandi(k_modes[[i]]$cluster,data_list_1[[i]]$groundTruth)
}

k_modes_plusOne_rand = NULL
set.seed(5)
for (i in seed_set) {
  k_modes_plusOne[[i]] = try(kmodes(data_list_1[[i]]$data,modes = (K+1)),TRUE)
  if(length(k_modes_plusOne[[i]])==1){
    k_modes_plusOne[[i]] = try(kmodes(data_list_1[[i]]$data,modes = (K+1)),TRUE)
  }
  if(length(k_modes_plusOne[[i]])>1){
    k_modes_plusOne_rand[i]=mcclust::arandi(k_modes_plusOne[[i]]$cluster,data_list_1[[i]]$groundTruth)
  }}

k_modes_minusOne_rand = NULL
set.seed(5)
for (i in seed_set) {
  k_modes_minusOne[[i]] = try(kmodes(data_list_1[[i]]$data,modes = (K-1)),TRUE)
  if(length(k_modes_minusOne[[i]])==1){
    k_modes_minusOne[[i]] = try(kmodes(data_list_1[[i]]$data,modes = (K-1)),TRUE)
  }
  if(length(k_modes_minusOne[[i]])>1){
    k_modes_minusOne_rand[i]=mcclust::arandi(k_modes_minusOne[[i]]$cluster,data_list_1[[i]]$groundTruth)
  }}

sim_1 = cbind(unlist(HD_rand),k_modes_minusOne_rand,k_modes_rand,k_modes_plusOne_rand,unlist(rand_VI_1))
colnames(sim_1) = c('HD','K-Modes[2]',"K-Modes[3]","K-Modes[4]",'HMM')

cat('Saving results \n');cat('\n')
save.image("output_1.RData")


