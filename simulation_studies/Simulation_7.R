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
# SIMULATION 7
#=========================================================================================

cat('Starting simulation set 7')

seed_set = 1:50

data_list_7 = list()
sim_7       = list()
psm_7       = list()
rand_VI_7   = list()

time = vector(mode='numeric',length=length(seed_set))

p = 150
K = 3
nj = 75
M.na.init = 1
k.init = K+1
gam = 0.2247071 #AntMAN::AM_find_gamma_Pois(n=sum(nj*K),Kstar=3,Lambda = 3)

for (seed in seed_set) {
  
  cat('Simulation 7.',seed,'\n')
  
  data_list_7[[seed]] = ham_mix_gen(M=c(3,4,5),
                                    k = K,
                                    p = p,
                                    n=c(nj,nj,nj),
                                    s=matrix(rep(0.2,K*p),ncol = p),
                                    seed = seed)
  
  set.seed(seed)
  C.init = sample(k.init,nj*K,replace = T)
  sigma.init = matrix(runif((k.init+M.na.init)*p),nrow = (k.init+M.na.init), ncol=p)
  cent.init =  matrix(sample(k.init,(k.init+M.na.init)*p,replace = T),nrow=(k.init+M.na.init),ncol = p)
  
  
  m = apply(data_list_7[[seed]]$data,2,function(x){length(table(x))})
  u=v=vector(length = p)
  
  u[m==3]=5.00
  v[m==3]=0.25
  
  u[m==4]=4.50
  v[m==4]=0.25
  
  u[m==5]=4.25
  v[m==5]=0.25
  
  set.seed(seed)
  start_time = Sys.time()
  sim_7[[seed]] = gibbs_mix_con(G=10000,
                                burnin = 5000,
                                data=data_list_7[[seed]]$data,
                                C.init = C.init,
                                k.init=k.init,
                                M.na.init = M.na.init,
                                cent.init = cent.init,
                                sigma.init = sigma.init,
                                u=u,
                                v=v, 
                                Lambda = 3,
                                gam = gam)
  end_time = Sys.time()
  time[seed] = end_time-start_time
  psm_7[[seed]] = comp.psm(sim_7[[seed]]$C)
  
  estim_7_VI_tmp  = mcclust.ext::minVI(psm_7[[seed]],method = 'all',cls.draw = sim_7[[seed]]$C)
  estim_7_VI = estim_7_VI_tmp$cl['best',]
  rand_VI_7[[seed]] = mcclust::arandi(estim_7_VI,data_list_7[[seed]]$groundTruth)

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
  HD[[i]] = CategorialCluster(data_list_7[[i]]$data)
  HD_rand[[i]]=mcclust::arandi(HD[[i]][[1]],data_list_7[[i]]$groundTruth)
}


#=========================================================================================
# K-MDOES
#=========================================================================================
cat('Starting K-MODES \n');cat('\n')

K=length(unique(data_list_7[[1]]$groundTruth))
k_modes = k_modes_plusOne = k_modes_minusOne = vector('list',length = 50)
k_modes_rand =  NULL
set.seed(5)
for (i in seed_set) {
  k_modes[[i]] = try(kmodes(data_list_7[[i]]$data,modes = K),TRUE)
  if(length(k_modes[[i]])==1){
    k_modes[[i]]=try(kmodes(data_list_7[[i]]$data,modes = K),TRUE)
  }
  if(length(k_modes[[i]])>1){
    k_modes_rand[i]=mcclust::arandi(k_modes[[i]]$cluster,data_list_7[[i]]$groundTruth)
  }}

k_modes_plusOne_rand = NULL
set.seed(5)
for (i in seed_set) {
  k_modes_plusOne[[i]] = try(kmodes(data_list_7[[i]]$data,modes = (K+1)),TRUE)
  if(length(k_modes_plusOne[[i]])==1){
    k_modes_plusOne[[i]] = try(kmodes(data_list_7[[i]]$data,modes = (K+1)),TRUE)
  }
  if(length(k_modes_plusOne[[i]])>1){
    k_modes_plusOne_rand[i]=mcclust::arandi(k_modes_plusOne[[i]]$cluster,data_list_7[[i]]$groundTruth)
  }}

k_modes_minusOne_rand = NULL
set.seed(5)
for (i in seed_set) {
  k_modes_minusOne[[i]] = try(kmodes(data_list_7[[i]]$data,modes = (K-1)),TRUE)
  if(length(k_modes_minusOne[[i]])==1){
    k_modes_minusOne[[i]] = try(kmodes(data_list_7[[i]]$data,modes = (K-1)),TRUE)
  }
  if(length(k_modes_minusOne[[i]])>1){
    k_modes_minusOne_rand[i]=mcclust::arandi(k_modes_minusOne[[i]]$cluster,data_list_7[[i]]$groundTruth)
  }}


sim_7 = cbind(unlist(HD_rand),k_modes_minusOne_rand,k_modes_rand,k_modes_plusOne_rand,unlist(rand_VI_7))
colnames(sim_7) = c('HD','K-Modes[2]',"K-Modes[3]","K-Modes[4]",'HMM')


cat('Saving results \n');cat('\n')
save.image("output_7.RData")


