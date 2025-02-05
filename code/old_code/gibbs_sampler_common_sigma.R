#Funzioni
source('../code/utility_functions.R',echo=T)

library(extraDistr) #---> for poisson mixture
library(bmixture) # ---> for gamma mixture 

#################
### ALGORITMO ###
#################

gibbs_ham=function(G,
                   burnin=0,
                   thin=1,
                   data,
                   eta,
                   Lambda,
                   M.init = 5,
                   M.max = 30,
                   gam=1,
                   a=2,
                   b=4){
  
  
  #############
  ### NOTES ###
  #############
  
  # - G          = number of values to be save (integer)
  # - thin       = thinning (integer, default = 1)
  # - burnin     = burnin (default = 0)
  # - data       = n*p matrix of data to cluster 
  # - eta        = standard deviation for MH-step proposal
  # - Lambda     = poisson mixuture hyperparameter
  # - gam        = weight of poisson mixture hyperparameter
  # - a          = gamma prior hyperparameter on sigma (alpha - shape)
  # - b          = gamma prior hyperparameter on sigma (beta - rate)
  # - M.na.init  = initial number of non-allocated components (integer, default = 1)
  # - M.max  = total number of M (components), M>M.max, algorithm will stop
  
  #Start timer
  start_time = proc.time()
  
  N = nrow(data)
  p = ncol(data)
  
  attri_list = attriList(data)
  attriSize = sapply(attri_list, length)
  
  #Initializing storage
  C      = matrix(NA, nrow = G+burnin, ncol = N)
  C.curr = c()
  
  M = c()
  k = k.curr = c()
  M.na = M.na.curr = c()
  M.curr = M.init
  
  U = c()
  
  Sm.init = rep(1,M.init)
  Sm      = list()
  Sm.curr = Sm.init
  t       = c()
  t.curr  = sum(Sm.curr)
  w       = matrix(0,nrow = G+burnin,ncol=M.max)
  
  
  #Initializing Hamming parameters
  
  p.Sigma = rep(1,M.init)
  p.Cent  = matrix(NA,nrow = M.init,ncol=p)
  
  for (i in 1:nrow(p.Cent)) {
    for (j in 1:ncol(p.Cent)) {
      p.Cent[i,j]=sample(attriSize,1,replace = F)
    }
  }
  
  Cent       = list()
  Cent.curr  = p.Cent
  
  Sigma      = list()
  Sigma.curr = p.Sigma
  
  #Initializing the lists
  for (i in 1:M.max) {
    Sm[[i]]    = c(NA)
    w[[i]]     = c(NA)
    Sigma[[i]] = c(NA)
    Cent[[i]]  = matrix(NA,nrow = G+burnin,ncol=p)
  }
  
  #Iterations:
  Iterations = burnin + (thin*G)
  g = 1
  pb <- txtProgressBar(min = 0, max = Iterations, style = 3)
  
  for (iter in 1:Iterations) {
    
    ###########################
    ### STEP A - SAMPLING U ###
    ###########################
    U.curr = rgamma(1,N,t.curr)
    if(iter<= burnin | iter%%thin == 0){
      U[g]=U.curr
    }
    ############################
    ### STEP B - SAMPLING ci ###
    ############################
    prob.m = prob.matrix(M=M.curr,N=N,Sm=Sm.curr,
                         Cent = Cent.curr,Sigma = Sigma.curr,
                         data = data,
                         attrisize = attriSize)
    for (i in 1:N) {
      C.curr[i]=sample(M.curr,1,replace = T,prob = prob.m[i,])
    }
    
    #Conto componenti allocate e non allocate 
    k.curr = length(unique(C.curr))
    
    #Rename allocated components from 1 to k
    for (i in 1:k.curr) {
      elements = which(C.curr==sort(unique(C.curr))[i])
      C.curr[elements]=i
    }
    
    #Salvo dati dello step
    if(iter<= burnin | iter%%thin == 0){
      C[g,] = C.curr
      k[g]  = k.curr
    }
    #############################
    ### STEP C.1 - SAMPLING M ###
    #############################
    lam = Lambda/((U.curr+1)^gam)
    w_1 = (((U.curr+1)^gam)*k.curr)/((((U.curr+1)^gam)*k.curr)+Lambda)
    w_2 = Lambda/((((U.curr+1)^gam)*k.curr)+Lambda)
    
    M.na.curr = rmixpois(1,lambda = c(lam,lam+1),alpha = c(w_1,w_2))
    
    M.curr = k.curr+M.na.curr 
    
    if(iter<= burnin | iter%%thin == 0){
      M[g]    = M.curr
      M.na[g]  = M.na.curr
    }
    ##########################################
    ### STEP C.2 - SAMPLING FROM ALLOCATED ###
    ##########################################
    
    ### Sm SAMPLING ###
    Sm.curr = c()
    for (i in 1:k.curr) {
      nm = sum(C.curr==i)
      Sm.curr[i] = rgamma(1,nm+gam,U.curr+1)
      if(iter<= burnin | iter%%thin == 0){
        Sm[[i]][g]=Sm.curr[i]
      }
    }
    ### CENTER SAMPLING ###
    Cent.curr = matrix(data=NA,nrow=M.curr,ncol = p)
    for (i in 1:k.curr) {
      prob.tmp = Center.prob(data =  data[C.curr==i,],sigma = Sigma.curr[i],
                             attrilist = attri_list,p=p)
      Cent.curr[i,] = Samp.Centr(prob_list = prob.tmp,attrilist = attri_list)
      if(iter<= burnin | iter%%thin == 0){
        Cent[[i]][g,]=Cent.curr[i,]
      }
    }
    ### SIGMA SAMPLING - MH ###
    
    #1.Computing distance from centers
    h_dist=list()
    for (i in 1:k.curr) {
      
      #Initializing vector
      h_dist[[i]]=vector('numeric',length = nrow(matrix(data[C.curr==i,],ncol=p)))
      
      for (j in 1:nrow(matrix(data[C.curr==i,],ncol=p))) {
        #Computing hamming distance for each allocated element
        h_dist[[i]][j] = dist_cate(a=matrix(data[C.curr==i,],ncol=p)[j,],
                                   b=Cent.curr[i,])
      }
    }
    #2.MH step for each allocated component
    for (i in 1:k.curr) {
      #Proposal
      s_prop = rnorm(1,Sigma.curr[i],eta[i])
      
      #Log-alpha
      if(s_prop<=0){
        log_alpha = -Inf
      }else{
        log_alpha = min(0, log_target(data=matrix(data[C.curr==i,],ncol=p),
                                      C=Cent.curr[i,],
                                      sigma=s_prop,
                                      a=a,
                                      b=b,
                                      attri_size = attriSize, 
                                      h_dist = h_dist[[i]])-
                          log_target(data=matrix(data[C.curr==i,],ncol=p),
                                     C=Cent.curr[i,],
                                     sigma = Sigma.curr[i],
                                     a=a,
                                     b=b,
                                     attri_size = attriSize,
                                     h_dist = h_dist[[i]]))
      }
      u=runif(1)
      if(log(u)<log_alpha){
        #ACCEPTING THE MOVE
        Sigma.curr[i]=s_prop
      }
      if(iter<= burnin | iter%%thin == 0){
        #Saving data
        Sigma[[i]][g]=Sigma.curr[i]
      }
    }
    #############################################
    ### STEP D.2 - SAMPLING FOR NON-ALLOCATED ###
    #############################################
    M.na.vector = 1:M.na.curr
    M.na.vector = M.na.vector+k.curr
    
    for (i in M.na.vector) {
      if(M.na.curr==0){
        break
      }else{
        
        ### SAMPLING Sm ###
        Sm.curr[i]=rgamma(1,gam,U.curr+1)
        if(iter<= burnin | iter%%thin == 0){
          Sm[[i]][g]=Sm.curr[i]
        }
        
        ### SAMPLING CENT ###
        for (j in 1:p) {
          Cent.curr[i,j]= sample(attri_list[[j]],1,
                                 replace = T)
        }
        if(iter<= burnin | iter%%thin == 0){
          Cent[[i]][g,]=Cent.curr[i,]
        }
        ### SAMPLING SIGMA ###
        Sigma.curr[i]=rgamma(1,a,b)
        if(iter<= burnin | iter%%thin == 0){
          Sigma[[i]][g]=Sigma.curr[i]
        }
      }
    }

    ##################
    ### UPDATING T ###
    ##################
    t.curr = sum(Sm.curr)
    w.curr = Sm.curr/t.curr
    if(iter<=burnin | iter%%thin == 0){
      t[g]=t.curr
      for (i in 1:length(Sm.curr)) {
        w[g,i]=w.curr[i]
      }
    }
    
    ######################################
    ### MOVING COUNTER FOR SAVING DATA ###
    ######################################
    if(iter<=burnin | iter%%thin == 0){
      g=g+1
    }
    
    #########################
    ### ITERATION COUNTER ###
    #########################
    #if(IT.counter.quater == T){
    #  it.count = iter/Iterations
    #  if(it.count%%0.25==0){
    #    cat(it.count*100,'% completed \n')
    #  }
    #}
    #if(IT.counter==T){
    #  it.count = iter/Iterations
    #  if((it.count)*100 %% 100 == 0){
    #    cat(it.count*100,'% completed k = ',k.curr,'\n')
    #  }
    #}
    
    setTxtProgressBar(pb, iter)
  }
  close(pb)
  
  #####################################
  ### REMOVING UNNECCESSARY VECTORS ###
  #####################################
  #res.Sm    = list()
  #res.Sigma = list()
  #res.Cent  = list()
  
  #for (i in 1:max(M)) {
  #  res.Sm[[i]]    = Sm[[i]]
  #  res.Sm[[i]]    = res.Sm[[i]][!is.na(res.Sm[[i]])]
    
  #  res.Sigma[[i]] = Sigma[[i]]
  #  res.Sigma[[i]] = res.Sigma[[i]][!is.na(res.Sigma[[i]])]
    
  #  res.Cent[[i]]  = Cent[[i]]
  #  res.Cent[[i]]  = res.Cent[[i]][complete.cases(res.Cent[[i]]),]
    
  #}
  
  ######################
  ### EXECUTION TIME ###
  ######################
  end_time       = proc.time()
  execution_time = end_time-start_time
  cat('\n')
  cat('Execution time:','\n')
  print(execution_time)
  
  ##########################
  ### COLLECTING RESULTS ###
  ##########################
  
  results           = list()
  results$w         = w
  #results$res.Sm    = res.Sm
  results$Sm        = Sm
  #results$res.Sigma = res.Sigma
  results$Sigma     = Sigma
  #results$res.Cent  = res.Cent
  results$Cent      = Cent
  results$k         = k 
  results$M         = M
  results$M.na      = M.na
  results$C         = C
  results$U         = U 
  results$Lambda    = Lambda
  
  return(results)
}
