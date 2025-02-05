ham_mix_gen= function(M,k,p,n,s,seed=10091995){
  
  #############
  ### NOTES ###
  #############
  
  # - M = VECTOR of random length on which the single m_j are sampled 
  # - k = number (non-empty) of components 
  # - p = number of variables 
  # - n = VECTOR of length p, number of rows per component, e.g: n=c(50,20)
  # - s = MATRIX k*p, scale parameter for each variable and for each component
  
  ##########################################################
  ### FUNCTIONS TO GENERATE MIXTURE FROM HAMMING DISTRIB ###
  ##########################################################
  
  #ATTRIBUTES GENERATIONS
  attri = function(M,p,seed = 10091995){
    set.seed(seed)
    attri_size = sample(M,p,replace=T)
    attri_list = list()
    for (i in 1:length(attri_size)) {
      attri_list[[i]]=1:attri_size[i]
    }
    results = list()
    results$attri_size = attri_size
    results$attri_list = attri_list
    return(results)
  }
  
  #FUNCTION TO SAMPLE CENTER
  Cent.gen = function(k,p,attri,seed=10091995){
    attri_list = attri$attri_list
    Cent = matrix(NA,nrow = k,ncol = p)
    set.seed(seed)
    for (i in 1:k) {
      for (j in 1:p) {
        Cent[i,j]=sample(attri_list[[j]],1)
      }
    }
    return(Cent)
  }
  
  
  #FUNCTION TO GENERATE ONE SINGLE CLUSTER
  data.clust = function(s_clust,n_clust,p,Cent_clust,attri,seed=10091995){
    attri_list = attri$attri_list
    attri_size = attri$attri_size
    
    #Recalling function prob.c to compute probabilities of center 
    prob.c_j = function(m_j,sigma_j){
      out= (m_j-1)/exp(1/sigma_j)
      return((1+out)^-1)
    }
    
    #Data generation
    data=matrix(NA,nrow = n_clust,ncol = p)
    set.seed(10091995)
    for (i in 1:n_clust) {
      for (j in 1:p) {
        prob=c(rep(NA,attri_size[j]))
        
        #P(a_j=c_j)
        c_prob = prob.c_j(m_j=attri_size[j],sigma_j=s_clust[j])
        prob[Cent_clust[j]==attri_list[[j]]]=c_prob
        
        #P(a_j≠c_j)
        prob[is.na(prob)]=(1-c_prob)/(attri_size[j]-1)
        
        #Sampling
        data[i,j]=sample(attri_list[[j]],1,prob=prob)
      }
    }
    return(data)
  }
  
  #######################
  ### DATA GENERATION ### 
  #######################
  
  attri.list  = attri(M=M,p=p,seed = seed)
  trueCent    = Cent.gen(k=k,p=p,attri=attri.list,seed = seed)
  groundTruth = c()
  
  data=c()
  for (i in 1:k) {
    s.tmp = s[i,]
    data  = rbind(data,data.clust(n_clust=n[i],s_clust=s.tmp,
                                  p=p,
                                  Cent_clust =trueCent[i,],
                                  attri = attri.list))
    
    groundTruth = append(groundTruth,rep(i,n[i]))
  }
  
  results             = list()
  results$data        = data
  results$groundTruth = groundTruth
  results$trueCent    = trueCent
  results$attributes  = attri.list
  return(results)
}