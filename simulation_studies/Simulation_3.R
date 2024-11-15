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
# SIMULATION 3
#=========================================================================================
seed_set = 1:50

cat('Starting simulation set 3')

data_list_3 = list()
sim_3       = list()
psm_3       = list()
rand_VI_3   = list()

M.na.init = 1
k.init = 2
p = 10
nj = 75
K = 4


for (seed in seed_set) {
  
  cat('Simulation 3.',seed,'\n')

  data_list_3[[seed]] = ham_mix_gen(M=c(3,4,5),
                                    k = K,
                                    p = p,
                                    n=c(nj,nj,nj,nj),
                                    s=matrix(rep(0.5,K*p),ncol = p),
                                    seed = seed)
  
  set.seed(seed)
  C.init = sample(k.init,nj*K,replace = T)
  sigma.init = matrix(runif((k.init+M.na.init)*p),nrow = (k.init+M.na.init), ncol=p)
  cent.init =  matrix(sample(k.init,(k.init+M.na.init)*p,replace = T),nrow=(k.init+M.na.init),ncol = p)
  
  m = apply(data_list_3[[seed]]$data,2,function(x){length(table(x))})
  u=v=vector(length = p)
  
  u[m==3]=5.00
  v[m==3]=0.25
  
  u[m==4]=4.50
  v[m==4]=0.25
  
  u[m==5]=4.25
  v[m==5]=0.25
  
  sim_3[[seed]] = gibbs_mix_con(G=10000,
                                burnin=5000,
                                data=data_list_3[[seed]]$data,
                                u=u,
                                v=v,
                                C.init = C.init,
                                k.init = k.init,
                                M.na.init = M.na.init,
                                cent.init = cent.init,
                                sigma.init = sigma.init,
                                Lambda = 4,
                                gam = 0.2833003)
  
  psm_3[[seed]] = comp.psm(sim_3[[seed]]$C)
  
  estim_3_VI_tmp  = mcclust.ext::minVI(psm_3[[seed]],method = 'all',cls.draw = sim_3[[seed]]$C)
  estim_3_VI = estim_3_VI_tmp$cl['best',]
  rand_VI_3[[seed]] = mcclust::arandi(estim_3_VI,data_list_3[[seed]]$groundTruth)
}



#=========================================================================================
# HD_vector
#=========================================================================================
cat('Starting HD-vector \n');cat('\n')

HD=list()
HD_rand = list()
set.seed(5)
for (i in seed_set) {
  HD[[i]] = CategorialCluster(data_list_3[[i]]$data)
  print(i)
  HD_rand[[i]]=mcclust::arandi(HD[[i]][[1]],data_list_3[[seed]]$groundTruth)
}


#=========================================================================================
# K-MDOES
#=========================================================================================
cat('Starting K-MODES \n');cat('\n')

K=length(unique(data_list_3[[1]]$groundTruth))
k_modes = k_modes_plusOne = k_modes_minusOne = vector('list',length = 50)
k_modes_rand =  NULL
set.seed(5)
for (i in seed_set) {
  k_modes[[i]] = kmodes(data_list_3[[i]]$data,modes = K)
  k_modes_rand[i]=mcclust::arandi(k_modes[[i]]$cluster,data_list_3[[i]]$groundTruth)
}

k_modes_plusOne_rand = NULL
set.seed(5)
for (i in seed_set) {
  k_modes_plusOne[[i]] = try(kmodes(data_list_3[[i]]$data,modes = (K+1)),TRUE)
  if(length(k_modes_plusOne[[i]])==1){
    k_modes_plusOne[[i]] = try(kmodes(data_list_3[[i]]$data,modes = (K+1)),TRUE)
  }
  if(length(k_modes_plusOne[[i]])>1){
    k_modes_plusOne_rand[i]=mcclust::arandi(k_modes_plusOne[[i]]$cluster,data_list_3[[i]]$groundTruth)
  }}

k_modes_minusOne_rand = NULL
set.seed(5)
for (i in seed_set) {
  k_modes_minusOne[[i]] = try(kmodes(data_list_3[[i]]$data,modes = (K-1)),TRUE)
  if(length(k_modes_minusOne[[i]])==1){
    k_modes_minusOne[[i]] = try(kmodes(data_list_3[[i]]$data,modes = (K-1)),TRUE)
  }
  if(length(k_modes_minusOne[[i]])>1){
    k_modes_minusOne_rand[i]=mcclust::arandi(k_modes_minusOne[[i]]$cluster,data_list_3[[i]]$groundTruth)
  }}


sim_3 = cbind(unlist(HD_rand),k_modes_minusOne_rand,k_modes_rand,k_modes_plusOne_rand,unlist(rand_VI_3))
colnames(sim_3) = c('HD',
                    paste('K-Modes[',(K-1),']',sep=""),
                    paste('K-Modes[',K,']',sep=""),
                    paste('K-Modes[',(K+1),']',sep=""),
                    'HMM')


cat('Saving results \n');cat('\n')
save.image("output_3.RData")


