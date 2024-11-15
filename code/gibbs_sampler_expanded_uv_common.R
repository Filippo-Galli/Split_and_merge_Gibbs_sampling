require(Rcpp)
require(extraDistr)

Rcpp::sourceCpp('../code/gibbs_utility.cpp')
Rcpp::sourceCpp('../code/hyperg.cpp')

gibbs_mix_con = function(G,
                         thin       = 1,
                         burnin     = 0,
                         data,
                         a1,
                         b1,
                         a2,
                         b2,
                         u_sigma.init,
                         v_sigma.init,
                         Lambda     = 1,
                         gam        = 1,
                         C.init     = NULL,
                         k.init     = 2, 
                         M.na.init  = 1,
                         cent.init  = NULL,
                         sigma.init = NULL,
                         M.max      = 50){
  
  #############
  ### NOTES ###
  #############
  
  # - G          = number of values to be save (integer)
  # - thin       = thinning (integer, default = 1)
  # - burnin     = burnin (default = 0)
  # - data       = n*p matrix of data to cluster 
  # - u          = sigma first hyperparameter
  # - v          = sigma second hyperparameter
  # - Lambda     = poisson mixuture hyperparameter
  # - gam        = weight of poisson mixture hyperparameter
  # - C.init     = initial allocation vector (vector, if not given it will be randomly generated within k.init)
  # - k.init     = initial number of allocated components (integer, default = 2)
  # - M.na.init  = initial number of non-allocated components (integer, default = 1)
  # - cent.init  = center parameter starting condition (matrix of size (k.init+M.na.init)*p, if not given, it willl be randomly generated)
  # - sigma.init = scale (sigma) parameter starting condition (matrix of size (k.init+M.na.init)*p, if not given, it willl be randomly generated)
  # - M.max  = total number of M (components), M>M.max, algorithm will stop
  
  n = nrow(data)
  p = ncol(data)
  
  data = as.matrix(data)
  
  #Fix total number of iterations
  Iterations = burnin+thin*G
  g          = 2
  
  #Attributes list and data 
  attri_List = Attributes_List(data=data,p=p)
  attri_size = sapply(attri_List, length)
  
  ###############################
  ### Initialize data storage ###
  ###############################
  
  M     = c()
  k     = c()
  M.na  = c()
  C     = matrix(NA, nrow = G+g, ncol = n)  
  U     = c()
  Sm    = list()
  w     = matrix(0,nrow = G+g,ncol=M.max)
  Cent  = list()
  Sigma = list()
  
  u_sigma = list()
  v_sigma = list()
  
  for (i in 1:M.max) {
    Cent[[i]]  = Sigma[[i]] = matrix(nrow =1,ncol=p)
  }
  
  for (i in 1:(G+g)) {
    Sm[[i]]    = c(NA)
    w[[i]]     = c(NA)
  }
  
  ######################
  ### Initial values ###
  ######################
  
  #Randomly generate initial allocation vector if not given
  if(is.null(C.init)){
    cat('Initial allocations not defined \n')
    C.init = sample(1:k.init,n,replace = T)
    cat('Sampling random allocations as starting conditions \n')
    cat('\n')
  }
  
  C[1,]      = C.init
  k[1]       = k.init = length(unique(C.init))
  M.na[1]    = M.na.init
  M[1]       = M.init = M.na.init + k.init
  U[1]       = U.init = 1
  Sm.init    = c(rep(1,k.init),rep(0,M.na.init))
  t.init     = sum(Sm.init)
  
  #Randomly generate starting conditions for Center and sigma parameters if not given
  if(is.null(cent.init)&&is.null(sigma.init)){
    cat('Initial center and scale (sigma) not defined \n')
    p.Sigma = matrix(c(rep(1,k.init+M.na.init)),
                     nrow = k.init+M.na.init,
                     ncol = p,
                     byrow = T)
    
    p.Cent  = matrix(NA,nrow = k.init+M.na.init,ncol=p)
    
    for (i in 1:nrow(p.Cent)) {
      for (j in 1:ncol(p.Cent)) {
        p.Cent[i,j]=sample(unlist(attri_List[j]),1,replace = F)
      }
    }
    
    Sigma.init = p.Sigma
    Cent.init  = p.Cent
    cat('Sampled random center and scale (sigma) as starting conditions \n')
    cat('\n')
  }else{
    Sigma.init = sigma.init
    Cent.init  = cent.init
  }
  
  pos_ref    = 1:k.init
  
  ### current values
  C.curr     = C.init
  k.curr     = k.init
  M.na.curr  = M.na.init
  M.curr     = M.init
  U.curr     = U.init
  Sm.curr    = Sm.init
  t.curr     = t.init
  Cent.curr  = Cent.init
  Sigma.curr = Sigma.init
  
  u_sigma.curr = u_sigma.init
  v_sigma.curr = v_sigma.init
  
  u_sigma[[1]] = u_sigma.init
  v_sigma[[1]] = v_sigma.init
  
  #SET PROGRESSBAR 
  cat('Start sampling... \n')
  cat('\n')
  pb = txtProgressBar(min = 0, max = Iterations, style = 3) 
  
  #===========#
  # ALGORITHM #
  #===========#
  
  for (iter in 2:Iterations) {

    ###############################
    ### SAMPLING FROM ALLOCATED ###
    ###############################
    Cent.curr = Sigma.curr.new = matrix(data=NA,nrow=M.curr,ncol = p)
    u_sigma.curr.new = v_sigma.curr.new = vector('numeric',length=M.curr)
    
     for (i in 1:k.curr) {
      
      data.tmp = matrix(data[C.curr==i,],ncol = p)
      
      ### Sm SAMPLING ###
      nm = sum(C.curr==i)
      Sm.curr[i] = rgamma(1,nm+gam,U.curr+1)


      ### SAMPLING CENTER ###
      prob.tmp = Center_prob(data=data.tmp,
                             sigma = Sigma.curr[pos_ref[i],],
                             attrisize = attri_size)
      
      Cent.curr[i,] = Samp_Center(center_prob = prob.tmp,attriList = attri_List,p=p)
      
      for (j in 1:p) {
        
        dd = sum(data.tmp[,j]!= Cent.curr[i,j])
        cc = nm-dd
        
        ### SAMPLING SIGMA ###
        Sigma.curr.new[i,j]=rhyper_sig(n = 1,
                                    d = v_sigma.curr[i]+dd,
                                    c = u_sigma.curr[i]+cc,
                                    m = attri_size[j])
      }
        ### SAMPLING u AND v ###
        
        ### sampling u
        proposal_u = rnorm(1,
                           mean=u_sigma.curr[i],
                           sd=0.5)
        if (proposal_u<=0){#if proposal is negative
          proposal_u = (-1)*proposal_u
        }
        
        #testing proposal for u 
        alpha_u = (sum(dhyper_sig_raf(x = Sigma.curr.new[i,],
                                  d = v_sigma.curr[i],
                                  c = proposal_u,
                                  m = attri_size[j],
                                  log_scale = T))+
                     dgamma(proposal_u,a1,b1,log = T))-
          (sum(dhyper_sig_raf(x = Sigma.curr.new[i,],
                          d = v_sigma.curr[i],
                          c = u_sigma.curr[i],
                          m = attri_size[j],
                          log_scale = T))+
             dgamma(u_sigma.curr[i],a1,b1,log = T))
        
        
        alpha_u <- min(0,alpha_u)
        test = log(runif(1))
        if(test<alpha_u){
          u_sigma.curr.new[i] = proposal_u
        }else{
          u_sigma.curr.new[i] = u_sigma.curr[i]
        }
        
        ### sampling v 
        proposal_v = rnorm(1,
                           mean=v_sigma.curr[i],
                           sd=0.5)
        if (proposal_v<=0){
          proposal_v = (-1)*proposal_v
        }
        
        alpha_v =  (sum(dhyper_sig_raf(x = Sigma.curr.new[i,],
                                   d = proposal_v,
                                   c = u_sigma.curr.new[i],
                                   m = attri_size[j],
                                   log_scale = T))+
                      dgamma(proposal_v,a2,b2,log = T))-
          (sum(dhyper_sig_raf(x = Sigma.curr.new[i,],
                          d = v_sigma.curr[i],
                          c = u_sigma.curr.new[i],
                          m = attri_size[j],
                          log_scale = T))+
             dgamma(v_sigma.curr[i],a2,b2,log = T))
    
        alpha_v = min(0,alpha_v)
        test = log(runif(1))
        if(test<alpha_v){
          v_sigma.curr.new[i] = proposal_v
        }else{
          v_sigma.curr.new[i] = v_sigma.curr[i]
        }
        
      
      
        
        
      }
    
    Sigma.curr = Sigma.curr.new
    u_sigma.curr = u_sigma.curr.new
    v_sigma.curr = v_sigma.curr.new
  
    ##################################
    ### SAMPLING FOR NON-ALLOCATED ###
    ##################################
    
    M.na.vector = 1:M.na.curr
    M.na.vector = M.na.vector+k.curr
    
    for (i in M.na.vector) {
      if(M.na.curr==0){
        break
      }else{

        ### SAMPLING Sm ###
        Sm.curr[i]=rgamma(1,gam,U.curr+1)
        
        ### SAMPLING u AND v ###
        u_sigma.curr[i] = rgamma(1,a1,b1)
        v_sigma.curr[i] = rgamma(1,a2,b2)
        
        for (j in 1:p) {
          
          ### SAMPLING CENTER ###
          Cent.curr[i,j]= sample(attri_List[[j]],1,
                                 replace = T)
          ### SAMPLING SIGMA ###
          Sigma.curr[i,j]=rhyper_sig(n = 1,
                                  d = v_sigma.curr[i],
                                  c = u_sigma.curr[i], 
                                  m = attri_size[j])
          
        }
      }
    }
  
    ############################
    ### SAMPLING ALLOCATIONS ###
    ############################
  
    prob.matrix = alloc_matrix(cent      = Cent.curr,
                               sigma     = Sigma.curr,
                               data      = data, 
                               M_curr    = M.curr,
                               attrisize = attri_size,
                               Sm        = Sm.curr,
                               log_scale = TRUE)
    
    prob.matrix = sweep(prob.matrix,1,apply(prob.matrix,1,max))
    prob.matrix = exp(prob.matrix)
    t           = rowSums(prob.matrix)
    prob.matrix = prob.matrix/t
    
    for (i in 1:n) {
      C.curr[i]=sample(M.curr,1,replace = T,prob = prob.matrix[i,])
    }
    
    #Counting allocated and non-allocated components
    sampled_Ci = unique(C.curr)
    sorted_Ci  = sort(sampled_Ci)
    k.curr     = length(sampled_Ci)
    
    #Rename allocated components from 1 to k
    pos_ref = NULL
    for (i in 1:k.curr) {
      elements         = which(C.curr==sorted_Ci[i])
      C.curr[elements] = i
      pos_ref[i]       = sorted_Ci[i]
    }
    
    ##################
    ### SAMPLING M ###
    ##################
    
    lam = Lambda/((U.curr+1)^gam)
    w_1 = (((U.curr+1)^gam)*k.curr)/((((U.curr+1)^gam)*k.curr)+Lambda)
    w_2 = Lambda/((((U.curr+1)^gam)*k.curr)+Lambda)
    
    M.na.curr = rmixpois(1,lambda = c(lam,lam+1),alpha = c(w_1,w_2))
    
    # total number of components
    M.curr = k.curr+M.na.curr 

    ##################
    ### UPDATING T ###
    ##################
    t.curr = sum(Sm.curr)
    w.curr = Sm.curr/t.curr

    ##################
    ### SAMPLING U ###
    ##################
    
    U.curr = rgamma(1,n,t.curr)
   
    #####################################
    ### MOVING COUNTER + SAVING DATA  ###
    #####################################
    
    if(iter>=burnin& iter%%thin == 0){
      
      U[g]       = U.curr
      C[g,]      = C.curr
      k[g]       = k.curr
      M[g]       = M.curr
      M.na[g]    = M.na.curr
      
      u_sigma[[g]] = u_sigma.curr
      v_sigma[[g]] = v_sigma.curr
      
      #Saving the Hamming parameters:
      for (i in 1:k.curr) {
        Cent[[i]]  = rbind(Cent[[i]],Cent.curr[i,])
        Sigma[[i]] = rbind(Sigma[[i]],Sigma.curr[i,])
        Sm[[g]][i] = Sm.curr[i]
      }
      t[g]       = t.curr
      
      for (i in 1:length(Sm.curr)) {
        w[g,i]   = w.curr[i]
      }
      g=g+1
    }
    setTxtProgressBar(pb, iter)
  }
  close(pb)
  
  ##########################
  ### COLLECTING RESULTS ###
  ##########################
  
  results           = list()
  results$w         = w
  results$Sm        = Sm
  results$Sigma     = Sigma
  results$Cent      = Cent
  results$k         = k 
  results$M         = M
  results$M.na      = M.na
  results$C         = C
  results$U         = U 
  results$Lambda    = Lambda
  results$u_sigma   = u_sigma
  results$v_sigma   = v_sigma
  
  return(results)
}
 