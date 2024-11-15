#=========================================================================================
# Soybean
#=========================================================================================
library(mcclust)
library(mcclust.ext)
library(AntMAN)
library(klaR)
source("../code/complement_functions.R")

#=========================================================================================
# Data loading and cleaning
#=========================================================================================
soybean=read.table("../data/soybean-small.data",h=F,sep=",")
groundTruth = as.numeric(substr(soybean$V36,start=2,stop=2))
classes = factor(groundTruth,labels=c("diaphorite stem canker","charcoal rot", 
"rhizoctonia root rot", "phytophthora rot"))
soy = soybean[,-36]


## multiple correspondence analysis
soy.f = as.data.frame(matrix(NA,nrow(soy),ncol(soy)))
for(i in 1:ncol(soy)){
  soy.f[,i] = as.factor(soy[,i])
}
acm = mca(soy.f,abbrev = TRUE)

# correspondence analysis
x11()
par(mar=c(3,3,1,1))
plot(predict(acm,soy.f ),col=groundTruth ,
     pch=c(17 ,15 ,18, 19, 23, 25, 8 )[groundTruth],
     xlab='',ylab='',main="")
legend("right",legend=unique(classes),col=unique(classes),
       pch=unique(c(17 ,15 ,18, 19, 23, 25, 8 )[groundTruth]),
       cex=0.9)
abline(h=0,lty=2)
abline(v=0,lty=2)

### as matrix
soy = as.matrix(soy+1)
mm = c(as.matrix(read.table("../data/soybean_mm.txt", quote="\"", comment.char=""))) # different from the observed
p = ncol(soy)
n = nrow(soy)

#=========================================================================================
# Gibbs sampler HMM common sigma
#=========================================================================================
source('../code/gibbs_sampler_common_sigma.R',echo=T)


Lambda = 4
gam = AntMAN::AM_find_gamma_Pois(n=nrow(soy), Lambda=Lambda,Kstar=4)
prior = AM_prior_K_Pois(n=nrow(soy), gam, Lambda = Lambda)

set.seed(10)
gibbs.soy = gibbs_ham(G = 10000,
                      burnin = 2000,
                      thin = 1,
                      data = soy,
                      eta = c(rep(0.2,30)),
                      gam =  gam,
                      Lambda = Lambda,
                      a=2,
                      b=4)

# posterior K
post_k = table(gibbs.soy$k)/length(1:10000)

## posterior similarity matrix
psm = comp.psm(gibbs.soy$C)

# Figure prior and posterior of K
xl=7
x11()
par(mar=c(3.5,2,1,1),mgp=c(2,1,0))
plot(post_k,lwd = 2,
     xlab = "k", ylab="", xlim=c(1,xl),axes=F)
segments(1:xl,rep(0,xl),1:xl,prior,col="red",pch=4)
axis(1,1:xl,1:xl,cex.axis=1)
axis(2)
legend("topleft",legend=c("P(K = k)","P(K = k | data)"),
       col=c("red",1),lwd=c(1,2))


# Figure posterior similarity matrix
x11()
par(mar=c(2.5,2.5,1,1),mgp=c(2,1,0))
myplotpsm(psm)


# estimated clusters
VI = minVI(psm)
table(VI$cl)

# adjusted Rand index
arandi(VI$cl,groundTruth)


#=========================================================================================
# Gibbs sampler HMM
#=========================================================================================
source('../code/gibbs_sampler.R', echo=TRUE)

# true number of clusters
Kstar = 4

# prior distribution
Lambda = 4
gam = AntMAN::AM_find_gamma_Pois(n=nrow(soy), Lambda=Lambda,Kstar=2)
prior = AM_prior_K_Pois(n=nrow(soy), gam, Lambda = Lambda)
plot(prior)

#m = c(2,    3,    4,    5,    6,    7)
uu = c(6,    5,    4.5,  4.25, 3,    3)
vv = c(0.25, 0.25, 0.25, 0.25, 0.25, 0.25)

u_soy = v_soy = NULL
for (i in 1:length(mm)){
  u_soy[i] = uu[mm[i]-1]
  v_soy[i] = vv[mm[i]-1]
}


# initial values
k.init = 10
M.na.init = 3
set.seed(20)
C.init = sample(1:k.init,n,replace=T)
cent.init = matrix(NA,nrow = (k.init+M.na.init), ncol=p)
for(i in 1:p){
  cent.init[,i] = sample(1:mm[i],k.init+M.na.init,replace=T) 
}
sigma.init = matrix(0.5,nrow = (k.init+M.na.init), ncol=p)


set.seed(57)
sim_soy = gibbs_mix_con(G=20000,
                          burnin = 10000,
                          data=soy,
                          u=u_soy,v=v_soy,
                          Lambda = Lambda,
                          gam = gam,
                          C.init = C.init, 
                          cent.init = cent.init,
                          sigma.init = sigma.init,
                          M.na.init = M.na.init,
                          M.max=30)

# posterior K
post_k = table(sim_soy$k[2:20002])/length(2:20002)

## posterior similarity matrix
psm = comp.psm(sim_soy$C[2:20002,])

# estimated clustering
VI = minVI(psm)
table(VI$cl)
arandi(VI$cl,groundTruth) 

# Figure S14a
xl=7
x11()
par(mar=c(3.5,2,1,1),mgp=c(2,1,0))
plot(post_k,lwd = 2,
     xlab = "k", ylab="", xlim=c(1,xl),axes=F)
segments(1:xl,rep(0,xl),1:xl,prior,col="red",pch=4)
axis(1,1:xl,1:xl,cex.axis=1)
axis(2)
legend("topleft",legend=c("P(K = k)","P(K = k | data)"),
       col=c("red",1),lwd=c(1,2))



# Figure S14b
x11()
par(mar=c(2.5,2.5,1,1),mgp=c(2,1,0))
myplotpsm(psm,classes = VI$cl, ax = T, ay = T)




 



#=========================================================================================
# competitors
#=========================================================================================
source('../code/competitors_functions.R')


### K-modes algorithm ####
k_mod_rand4 = NULL
set.seed(18)
for(i in 1:100){
  kmodes_cluster4 = kmodes(soy,4)$cluster
  k_mod_rand4[i] = arandi(kmodes_cluster4,groundTruth)
}
mean(k_mod_rand4)
sd(k_mod_rand4)

### HD-vector algorithm ###
HD_rand = NULL
set.seed(1185)
for(i in 1:100){
  HD_output = CategorialCluster(soy)[[1]]
  HD_rand[i] = arandi(HD_output,groundTruth)
}
mean(HD_rand)
sd(HD_rand)


####### silhoutte index ######
library(cluster)
dist_mat = matrix(NA,nrow=nrow(soy),ncol = nrow(soy))

for (i in 1:nrow(soy)){
  for (j in 1:nrow(soy)){
    dist_mat[i,j] = hamming_distance(soy[i,],soy[j,])
  }
}

# Figure S15
x11()
par(mfrow=c(1,2))
sil_vi = silhouette(VI$cl,dmatrix = dist_mat)
plot(sil_vi, main='HMM')
sil_hd = silhouette(HD_output,dmatrix = dist_mat)
plot(sil_hd,main='HD')

