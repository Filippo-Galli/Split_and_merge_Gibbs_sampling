library(Rcpp)
library(extraDistr) #---> per mistura di poisson 
library(bmixture) # ---> per mistura di gamma

Rcpp::sourceCpp('../code/gibbs_utility.cpp')
Rcpp::sourceCpp('../code/hyperg2.cpp')


#################
### NOTE .cpp ###
#################

#rhyper_sig2(n,d,c,m)
# - n = number of sampled valies 
# - m = m_j --> number of attributes/modealities for the j'th column
# - d = hamming distance (1-kronecher distance ---> n-sum(delta)) ---> VECTOR 
# - c = nrow data - d                                             ---> VECTOR

gibbs_param_estim = function(G,thin=1,burnin=0,data,u,v,Lambda=1,gam=1,
                             estim_Truth, cent.init, sigma.init, Sm.init){
  
  #############
  ### NOTES ###
  #############
  
  # - G      = number of desired SAVED values (NOT OVERALL ITERATIONS)
  # - thin   = thin (default = 1)
  # - burnin = burnin (default = 0)
  # - data   = n*p matrix of data to cluster 
  # - u      = sigma first hyperparameter
  # - v      = sigma second hyperparameter
  # - Lambda = poisson mixuture hyperparameter
  # - gam    = weight of poisson mixture hyperparameter

  n = nrow(data)
  p = ncol(data)
  data = as.matrix(data)
  k = length(unique(estim_Truth))
  
  #################
  ### ALGORITHM ###
  #################
  
  #Fix total number of iterations
  Iterations = burnin+thin*G
  g          = 2
  
  #Attributes list and data 
  attri_List  = Attributes_List(data=data,p=p)
  attri_size = sapply(attri_List, length)
  
  # storage
  #Initialize data storage
  cat('Initializing data storage \n')
  cat('\n')
  #M = c()
  #k = c()
  #M.na = c()
  #C      = matrix(NA, nrow = G+burnin, ncol = n)  
  U = NULL
  t = NULL
  Cent  = Sigma  = array(NA, dim=c((G+burnin), k, p))
  Sm  = w  = matrix(NA, nrow = G+burnin, ncol = k)

  
  ### initial values
  C.init = estim_Truth
  # k[1]  = k.init = length(unique(C.init))
  # M.na[1] = M.na.init
  # M[1]  = M.init = M.na.init + k.init
  U[1]  = U.init = 1
  Sm[1,] = Sm.init 
  t[1] = t.init  = sum(Sm.init)
  w[1,] = Sm[1,]/t[1]
  Sigma[1,,] = Sigma.init = sigma.init
  Cent[1,,] = Cent.init = cent.init
  indicivecchi = 1:k

  
  ### current values
  C.curr = C.init
  # k.curr = k.init
  # M.na.curr = M.na.init
  # M.curr = M.init
  U.curr = U.init
  Sm.curr = Sm.init
  t.curr = t.init
  Cent.curr  = Cent.init
  Sigma.curr = Sigma.init
  

  #SET PROGRESSBAR 
  cat('Start sampling... \n')
  cat('\n')
  pb = txtProgressBar(min = 0, max = Iterations, style = 3) 
  cat("\n")
  for (iter in 2:Iterations) {
 # cat("iter = ", iter, "\n")
      
    

    ##########################################
    ### STEP C.2 - SAMPLING FROM ALLOCATED ###
    ##########################################
    Cent.curr = Sigma.curr.new = matrix(data=NA,nrow=k,ncol = p)
    
     for (i in 1:k) {
      
      data.tmp = matrix(data[C.curr==i,],ncol = p)
      
      ### Sm SAMPLING ###
      nm = sum(C.curr==i)
      Sm.curr[i] = rgamma(1,nm+gam,U.curr+1)


      ### SAMPLING CENTER ###
      prob.tmp = Center_prob(data=data.tmp,
                             sigma = Sigma.curr[indicivecchi[i],],
                             attrisize = attri_size)
      
      Cent.curr[i,] = Samp_Center(center_prob = prob.tmp,attriList = attri_List,p=p)
      
      
      # image(t(matrix(t(Cent.curr[i,]),ncol =16, byrow=TRUE)[16:1 ,]),col=gray (255:0/255) ,axes=F); box()
      # cat("iter=", iter, "\n clustr=",i, "\n")
      # readline("ciao")
      # 
      for (j in 1:p) {
        
        dd = sum(data.tmp[,j]!= Cent.curr[i,j])
        cc = nm-dd
        
        ### SAMPLING SIGMA ###
        Sigma.curr.new[i,j]=rhyper_sig2(n = 1,
                                       d = v[j]+dd,#possibile errore --> d deve essere v, metnre c deve essere u
                                       c = u[j]+cc,
                                       m = attri_size[j])
      }
    }
    Sigma.curr = Sigma.curr.new
   # cat("finito allocate \n")
    
    
  

 
    ##################
    ### UPDATING T ###
    ##################
    t.curr = sum(Sm.curr)
    w.curr = Sm.curr/t.curr
   # cat("finito pesi \n", "pesi = ", Sm.curr)
    
    ###########################
    ### STEP A - SAMPLING U ###
    ###########################
    U.curr = rgamma(1,n,t.curr)
   # cat("finito u \n")
    
    #########################################################################
    ### MOVING COUNTER FOR SAVING DATA + SAVING DATA OF CURRENT ITERATION ###
    #########################################################################
    
    if(iter>=burnin| iter%%thin == 0){
      # C[g,]      = C.curr
      # k[g]       = k.curr
      # M[g]       = M.curr
      # M.na[g]    = M.na.curr
      U[g]       = U.curr

      #Saving the Hamming parameters:
      Cent[g,,]  = Cent.curr
      Sigma[g,,] = Sigma.curr
      t[g]       = t.curr
      Sm[g,]     = Sm.curr
      w[g,]      = w.curr

        
              g=g+1
    }
   # cat("finito salvataggio")
    
    setTxtProgressBar(pb, iter)
  }
  close(pb)
  
  # #Delete first row of Sigma and Cent list (NA row)
  # for (i in 1:M.max) {
  #   Cent[[i]]  = Cent[[i]][-1,]
  #   Sigma[[i]] = Sigma[[i]][-1,]
  # }
  
  ##########################
  ### COLLECTING RESULTS ###
  ##########################
  
  results           = list()
  results$w         = w
  results$Sm        = Sm
  results$Sigma     = Sigma
  results$Cent      = Cent
  results$U         = U 
 
  return(results)
}
