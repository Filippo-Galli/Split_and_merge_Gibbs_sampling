#########################
### UTILITY FUNCTIONS ###
#########################

#HAMMING CATEGORICAL DISTANCE
dist_cate=function(a,b){
  distance = 0
  for (i in 1:length(a)){
    distance=distance+(a[i]!=b[i])
  }
  return(distance)
}

#ATTRIBUTES LIST
#attriList = function(mat){
#  ans = apply(mat,2,function(x) sort(unique(x)))
#  return(ans)
#}

attriList = function(mat){
  ans = list()
  for (i in 1:ncol(mat)) {
    ans[[i]]=sort(unique(mat[,i]))
  }
  return(ans)
  }

#HAMMING DENSITY
dhamming = function(x,C,s,attrisize){
  num = exp(-dist_cate(x,C)/s)
  den=c()
  for (i in 1:length(attrisize)) {
    den[i]=1+((attrisize[i]-1)/exp(1/s))
  }
  den = prod(den)
  out=num/den
  return(out)
}

#PROBABILITY MATRIX FOR SAMPLING Ci
prob.matrix = function(M,N,Sm,Cent,Sigma,data,attrisize){
  out= matrix(NA,ncol=M,nrow=N)
  for (i in 1:N) {
    for(j in 1:M){
      out[i,j]=Sm[j]*dhamming(x=data[i,],C=Cent[j,],s=Sigma[j],attrisize = attrisize)
    }
  }
  t=rowSums(out)
  out=out/t
  return(out)
}

#######################################
### FUNCTION FOR SAMPLNG PARAMETERS ###
#######################################

#CENTER PROBABILITIES
#Center.prob = function(data,sigma,attrisize,p){
#  prob = list()
#  data.tmp = matrix(data,ncol = p)
#  for (i in 1:ncol(data.tmp)) {
#    elements = as.vector(table(data.tmp[,i]))
#    prob_tmp = vector('numeric',length = attrisize[i])
#    for (j in 1:length(elements)) {
#      prob_tmp[j]=exp(-(nrow(data.tmp)-elements[j])/sigma)
#    }
#    prob_tmp=prob_tmp/sum(prob_tmp)
#    prob[[i]]=prob_tmp
#  }
#  return(prob)
#}
Center.prob = function(data,sigma,attrilist,p){
  prob = list()
  data.tmp = matrix(data,ncol = p)
  for (i in 1:ncol(data.tmp)) {
    elements = as.vector(table(data.tmp[,i]))
    pos      = as.numeric(names(table(data.tmp[,i])))
    prob_tmp = vector('numeric',length = max(attrilist[[i]]))
    prob_tmp[pos]=elements
    for (j in 1:length(prob_tmp)) {
      prob_tmp[j]=exp(-(nrow(data.tmp)-prob_tmp[j])/sigma)
    }
    prob_tmp=prob_tmp/sum(prob_tmp)
    prob[[i]]=prob_tmp
  }
  return(prob)
}


# CENTER SAMPLING
Samp.Centr = function(prob_list,attrilist){
  samp.C = c()
  for (i in 1:length(prob_list)) {
    if(length(prob_list[[i]])==1){
      samp.C[i]=attrilist[[i]]
    }else{
      samp.C[i]=sample(1:max(attrilist[[i]]),1,replace = F,prob=prob_list[[i]])
    }
  }
  return(samp.C)
}

# LOG-TARGET FOR SIGMA
log_target = function(data,C,sigma,a,b,attri_size,h_dist){
  n = nrow(data)
  out = -n*sum(log(1+((attri_size-1)/(exp(1/sigma)))))-(1/sigma)*sum(h_dist)-(b/sigma)-(a+1)*log(sigma)
  return(out)
}