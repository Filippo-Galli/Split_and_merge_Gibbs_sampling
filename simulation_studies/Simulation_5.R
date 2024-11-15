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

seed_set = 1:50

#=========================================================================================
# SIMULATION 5
#=========================================================================================

cat('Starting simulation set 5 \n')

data_list_5 = list()
sim_5       = list()
psm_5       = list()
rand_VI_5   = list()

time = vector(mode='numeric',length=length(seed_set))

M.na.init = 1
k.init = 2
p = 15
n = c(10,20,30)
K = 3
n_observations = sum(n)
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
  gam = 0.3028313 #Computed through AntMAN::AM_find_gamma_Pois(n=sum(5+20+50),Kstar=3,Lambda = 3)

  start_time = Sys.time()
  
  sim_5[[seed]] = gibbs_mix_con(G=10000,
                                burnin=5000,
                                data=data_list_5[[seed]]$data,
                                u=u,
                                v=v,
                                C.init = C.init,
                                k.init = k.init,
                                M.na.init = M.na.init,
                                cent.init = cent.init,
                                sigma.init = sigma.init,
                                Lambda = 3,
                                gam = gam)
  end_time = Sys.time()
  
  time[seed] = end_time-start_time
  
  psm_5[[seed]] = comp.psm(sim_5[[seed]]$C)
  
  estim_5_VI_tmp  = mcclust.ext::minVI(psm_5[[seed]],method = 'all',cls.draw = sim_5[[seed]]$C)
  estim_5_VI = estim_5_VI_tmp$cl['best',]
  rand_VI_5[[seed]] = mcclust::arandi(estim_5_VI,data_list_5[[seed]]$groundTruth)
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
# K-MODES
#=========================================================================================
cat('Starting K-MODES \n');cat('\n')

K=length(unique(data_list_5[[1]]$groundTruth))
k_modes = k_modes_plusOne = k_modes_minusOne = vector('list',length = 50)
k_modes_rand =  NULL
set.seed(15)
for (i in seed_set) {
  k_modes[[i]] = kmodes(data_list_5[[i]]$data,modes = K)
  k_modes_rand[i]=mcclust::arandi(k_modes[[i]]$cluster,data_list_5[[i]]$groundTruth)
}

k_modes_plusOne_rand = NULL
set.seed(15)
for (i in seed_set) {
  k_modes_plusOne[[i]] = try(kmodes(data_list_5[[i]]$data,modes = (K+1)),TRUE)
  if(length(k_modes_plusOne[[i]])==1){
    k_modes_plusOne[[i]] = try(kmodes(data_list_5[[i]]$data,modes = (K+1)),TRUE)
  }
  if(length(k_modes_plusOne[[i]])>1){
    k_modes_plusOne_rand[i]=mcclust::arandi(k_modes_plusOne[[i]]$cluster,data_list_5[[i]]$groundTruth)
  }}

k_modes_minusOne_rand = NULL
set.seed(15)
for (i in seed_set) {
  k_modes_minusOne[[i]] = try(kmodes(data_list_5[[i]]$data,modes = (K-1)),TRUE)
  if(length(k_modes_minusOne[[i]])==1){
    k_modes_minusOne[[i]] = try(kmodes(data_list_5[[i]]$data,modes = (K-1)),TRUE)
  }
  if(length(k_modes_minusOne[[i]])>1){
    k_modes_minusOne_rand[i]=mcclust::arandi(k_modes_minusOne[[i]]$cluster,data_list_5[[i]]$groundTruth)
  }}


rand_VI_5 = unlist(rand_VI_5)
sim_5 = cbind(unlist(HD_rand),k_modes_minusOne_rand,k_modes_rand,k_modes_plusOne_rand,rand_VI_5)
colnames(sim_5) = c('HD','K-Modes[2]',"K-Modes[3]","K-Modes[4]",'HMM')

cat('Saving results \n');cat('\n')

save.image("output_5.RData")




