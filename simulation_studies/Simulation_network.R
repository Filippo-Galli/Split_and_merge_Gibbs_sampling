#=========================================================================================
# NETWORK-BASED SIMULATION
#=========================================================================================
library(mcclust)
library(mcclust.ext)
library(klaR)
library(ggplot2)
library(GGally)
library(network)
library(sna)
library(FactoMineR)
library(factoextra)

cat('Loading functions \n');cat('\n')
source("../code/graph_generation.R")
source('../code/gibbs_sampler.R', echo=TRUE)
source('../code/competitors_functions.R')

cat('Functions correctly loaded \n');cat('\n')


#=========================================================================================
# SIMULATION graph
#=========================================================================================

seed_set = 1:50

data_list = list()
out       = list()
psm       = list()
rand_VI   = list()

time = vector(mode='numeric',length=length(seed_set))
m = 4
p = 15
K = 3
nj = 75
n = nj*K
M.na.init = 1
k.init = K+1
gam = 0.2247071 #AntMAN::AM_find_gamma_Pois(n=sum(nj*K),Kstar=3,Lambda = 3)

propp = matrix(c(0,0.5,0.55,0.75,1,
                 0,0.05,0.1,0.15,1,
                 0,0.05,0.1,0.95,1),m+1,K,byrow=F)

Khat = NULL

for (seed in seed_set) {
  cat('Network-based Simulation ',seed,'\n')
  
  Y = NULL
  Sigma = Graph = array(NA,dim=c(p,p,K))
  
  for (k in 1:K){

    # data generation
    graph_tmp = matrix(0, p, p)
    set.seed(k)
    graph_tmp[lower.tri(graph_tmp)] = sample(c(0,1), p*(p-1)/2, replace = TRUE, prob = c(1 - q, q))
    graph = graph_tmp + t(graph_tmp)

    data_temp = gen_UG_data(seed = seed, q = p, p = 0.1, n = nj, M = m, 
                            propp=propp[,k],graph=TRUE,true_graph=graph)
    Y = rbind(Y,data_temp$Y)
    Sigma[,,k] = data_temp$Sigma
    Graph[,,k] = data_temp$graph
  }
  data_list[[seed]] = list(Y = Y, groundTruth = rep(1:3,each=nj), Sigma = Sigma, Graph = Graph)


  #initial values
  C.init = sample(1:k.init,n,replace = T)
  sigma.init = matrix(0.5,nrow = (k.init+M.na.init), ncol=p)
  cent.init = matrix(data=1,nrow=(k.init+M.na.init),ncol = p)
  
  
  u=v=vector(length = p)
  
  u[m==3]=5.00
  v[m==3]=0.25
  
  u[m==4]=4.50
  v[m==4]=0.25
  
  u[m==5]=4.25
  v[m==5]=0.25
  
  set.seed(seed)
  out[[seed]] = gibbs_mix_con(G=12000,
                                burnin = 2000,
                                data=data_list[[seed]]$Y,
                                C.init = C.init,
                                M.na.init = M.na.init,
                                cent.init = cent.init,
                                sigma.init = sigma.init,
                                u=u,
                                v=v, 
                                Lambda = 3,
                                gam = gam)
  
  psm[[seed]] = comp.psm(out[[seed]]$C)
  
  estim_VI_tmp  = mcclust.ext::minVI(psm[[seed]],method = 'all',cls.draw = out[[seed]]$C)
  estim_VI = estim_VI_tmp$cl['best',]
  rand_VI[[seed]] = mcclust::arandi(estim_VI,data_list[[seed]]$groundTruth)
  
  # K estimate
  Khat[seed] = length(levels(as.factor(estim_VI)))
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
  HD[[i]] = CategorialCluster(data_list[[i]]$Y)
  HD_rand[[i]]=mcclust::arandi(HD[[i]][[1]],data_list[[i]]$groundTruth)
}


#=========================================================================================
# K-MDOES
#=========================================================================================
cat('Starting K-MODES \n');cat('\n')
k_modes = k_modes_plusOne = k_modes_minusOne = vector('list',length = 50)
k_modes_rand =  NULL
set.seed(5)
for (i in seed_set) {
  k_modes[[i]] = try(kmodes(data_list[[i]]$Y,modes = K),TRUE)
  if(length(k_modes[[i]])==1){
    k_modes[[i]] = try(kmodes(data_list[[i]]$Y,modes = K),TRUE)
  }
  if(length(k_modes[[i]])>1){
    k_modes_rand[i]=mcclust::arandi(k_modes[[i]]$cluster,data_list[[i]]$groundTruth)
  }
}

k_modes_plusOne_rand = NULL
set.seed(5)
for (i in seed_set) {
  k_modes_plusOne[[i]] = try(kmodes(data_list[[i]]$Y,modes = (K+1)),TRUE)
  if(length(k_modes_plusOne[[i]])==1){
    k_modes_plusOne[[i]] = try(kmodes(data_list[[i]]$Y,modes = (K+1)),TRUE)
  }
  if(length(k_modes_plusOne[[i]])>1){
    k_modes_plusOne_rand[i]=mcclust::arandi(k_modes_plusOne[[i]]$cluster,data_list[[i]]$groundTruth)
  }}

k_modes_minusOne_rand = NULL
set.seed(5)
for (i in seed_set) {
  k_modes_minusOne[[i]] = try(kmodes(data_list[[i]]$Y,modes = (K-1)),TRUE)
  if(length(k_modes_minusOne[[i]])==1){
    k_modes_minusOne[[i]] = try(kmodes(data_list[[i]]$Y,modes = (K-1)),TRUE)
  }
  if(length(k_modes_minusOne[[i]])>1){
    k_modes_minusOne_rand[i]=mcclust::arandi(k_modes_minusOne[[i]]$cluster,data_list[[i]]$groundTruth)
  }}


sim_graph = cbind(unlist(HD_rand),k_modes_minusOne_rand,k_modes_rand,k_modes_plusOne_rand,unlist(rand_VI))
colnames(sim_graph) = c('HD','K-Modes[2]',"K-Modes[3]","K-Modes[4]",'HMM')

cat('Saving outputs \n')
save.image(paste("output_network_",q,".RData",sep=""))


