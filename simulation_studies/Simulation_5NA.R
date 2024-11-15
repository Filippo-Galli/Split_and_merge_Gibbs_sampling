#=========================================================================================
# SIMULATION STUDY 
#=========================================================================================
library(mcclust)
library(mcclust.ext)
library(FactoMineR)
library(factoextra)
library(gridExtra)

cat('Loading functions \n');cat('\n')
source('../code/data_generation.R', echo=TRUE)
source('../code/gibbs_sampler_imputation.R', echo=TRUE)
cat('Functions correctly loaded \n');cat('\n')

seed_set = 1:50

#=========================================================================================
# SIMULATION 5 MISSING VALUE 
#=========================================================================================

cat('Starting simulation set 5 with missing values \n')

data_list_5 = list()
sim_5       = list()
psm_5       = list()
rand_VI_5   = list()

imputed_data = list()
true_values = list()

time = vector(mode='numeric',length=length(seed_set))

K = 3
p=15
n = c(10,20,30)
n_observations = sum(n)
sigma = 0.5
#p.missing = 0.1 # number of ones (10% of total elements)

M.na.init = 1
k.init = K + 1
gam = 0.3223624 #Computed through AntMAN::AM_find_gamma_Pois(n=n_observations,Kstar=K,Lambda = 3)

for (seed in seed_set) {
  
  cat('Simulation 5 imputation.',seed,'\n')
  
  
  
  data_list_5[[seed]] = ham_mix_gen(M=c(3,4,5),
                                    k = 3,
                                    p = p,
                                    n=n,
                                    s=matrix(rep(sigma,K*p),ncol = p),
                                    seed = seed)
  
  total_elements = n_observations*p  # total number of elements in the matrix
  num_ones = ceiling(p.missing * total_elements)  
  vec = c(rep(1, num_ones), rep(0, total_elements - num_ones))
  
  # Shuffle the vector to randomize the positions of 1s and 0s
  vec = sample(vec)
  # Convert the vector to a matrix
  mat_missing = matrix(vec, nrow=n_observations, ncol=p)
  imputation_matrix = 1-mat_missing
  
  #dropping values
  data_na = data_list_5[[seed]]$data*imputation_matrix
  
  #random values for beginning
  attributes_sizes = data_list_5[[1]]$attributes$attri_size
  
  n_missing_per_column = c()
  for (pos in 1:p){
    n_missing_per_column[pos] = sum(data_na[,pos]==0)
  }
  missing_pos = which(data_na==0,arr.ind = T)
  
  true_values_tmp = c()
  for (i in 1:nrow(missing_pos)){
    r = missing_pos[i,1]
    c = missing_pos[i,2]
    data_na[r,c] = sample(attributes_sizes[c],size=1)
    
    #storing true_values
    true_values_tmp[i] = data_list_5[[seed]]$data[r,c]
  }
  true_values[[seed]] = true_values_tmp
  
  #initializing parameters for gibbs sampler
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
  
  start_time = Sys.time()
  
  sim_5[[seed]] = gibbs_mix_con_imput(G=10000,
                                burnin=5000,
                                data_with_na = data_na,
                                missing_pos = missing_pos,
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
  
  imputed_data[[seed]] = sim_5[[seed]]$samp_data
  
  psm_5[[seed]] = comp.psm(sim_5[[seed]]$C)
  
  estim_5_VI_tmp  = mcclust.ext::minVI(psm_5[[seed]],method = 'all',cls.draw = sim_5[[seed]]$C)
  estim_5_VI = estim_5_VI_tmp$cl['best',]
  rand_VI_5[[seed]] = mcclust::arandi(estim_5_VI,data_list_5[[seed]]$groundTruth)
}


cat('Saving outputs \n')

save.image(paste("output_5NA_",p.missing*100,".RData",sep=""))


