library(BDgraph)
library(mvtnorm)


gen_UG_data = function(seed, q, n, M, propp, graph = F, p, true_graph){
  
  # i : seed for random generation
  # n : sample size
  # q : number of variables
  # M : number of categories (same for all q variables)
  # p = probability of edge inclusion
  # propp = proportion of each category
  
  # 1. DAG random generation
  #true_dag = randomDAG(q, prob = p)
  
  #A = as(true_dag, "matrix")
  
  ####
  {if (graph==TRUE){
    true_graph = true_graph
  }
  else
  {  
    # 1. UG random generation
    true_graph_tmp = matrix(0, q, q)
    true_graph_tmp[lower.tri(true_graph_tmp)] = sample(c(0,1), q*(q-1)/2, replace = TRUE, prob = c(1 - p, p))
    true_graph = true_graph_tmp + t(true_graph_tmp)
  }}
  ####
  
  # 2. Random parameter generation following the linear SEM (B and sigma_cond)
  
  #B = A*matrix(runif(q*q, 0.1, 1), q, q)*sample(c(-1, 1), size = q*q, replace = TRUE); diag(B) = 1
  
  #sigma_cond = diag(rep(1, q))
  
  #Sigma = solve(t(B))%*%sigma_cond%*%solve(B); 
   
  
  ####
  
  # 2. Random randomly generate Sigma (directly) from a G-Wishart using BDgraph
  set.seed (seed)
  Precision = rgwish( n = 1, adj = true_graph, b = 3, D = diag(q), threshold = 1e-8 ) 
  Sigma =  solve(Precision)
  
  ####
  
  # 3. Generate the latent (Gaussian) data
  
  library(mvtnorm)
  
  mu = c(rep(0, q))
  Z = data.frame(rmvnorm(n, mu, Sigma))
  
  # 4. Threshold the data such that 
  
  #Y[Y > 0] = 1
  #Y[Y < 0] = 0
  
  # More in general, generate theta (m,1) vector of cutoffs and set Y = {0,1,...,m} accordingly
  cutoff = apply(Z,2,quantile, prob=propp)
  #cutoff = qnorm(propp)
  
  Y = matrix(NA,dim(Z)[1],dim(Z)[2])
  for(j in 1:q){
    for(m in 2:(M+1)){
      Y[(Z[,j]>=cutoff[(m-1),j]&Z[,j]<=cutoff[m,j]),j] = m-1
    }}

  
  return(list(graph = true_graph, Y = Y, Z = as.matrix(Z) ,Sigma = Sigma))
  
}


## Example:

#data_graph = gen_UG_data(i = 1, q = 15, n = 100, M = 4, propp=seq(0,1,1/4), graph=T, true_graph=true_graph)
#data_graph = gen_UG_data(i = 1, q = 15, n = 100, M = 4, propp=seq(0,1,1/4), graph=F, p=0.25)



