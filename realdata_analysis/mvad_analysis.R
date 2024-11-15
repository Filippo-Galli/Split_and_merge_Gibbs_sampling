#install.packages("MEDseq")
library("MEDseq")
library(mcclust)
library(mcclust.ext)
library(AntMAN)
library(klaR)
library(TraMineR)
source('../code/gibbs_sampler2.R', echo=TRUE)
source('../code/complement_functions.R')



##################################################
########### mvad data ############################
##################################################

data(mvad, package="MEDseq")
mvad.seq = mvad[,17:86]
mvad.lab = c("employment", "FE", "HE",
             "joblessness", "school", "training")


mvad.data = matrix(0,nrow(mvad.seq),ncol(mvad.seq))
for(i in 1:length(mvad.lab)){
  mvad.data[mvad.seq==mvad.lab[i]]=i
}

mvad.LAB = c("Employment", "FE", "HE", "Joblessness", "School", "Training")
months=colnames(mvad)[15:86]

p = ncol(mvad.data)
n = nrow(mvad.data)
m = rep(6, p)

###########################################################################
 
# number of clusters a priori
Kstar = 5

# prior hyperparameters
Lambda = 5
gam = 0.2833003#AntMAN::AM_find_gamma_Pois(n=nrow(mvad.data),Lambda=Lambda,Kstar=Kstar)
prior = AM_prior_K_Pois(n=nrow(mvad), gam, Lambda = Lambda)
plot(prior)

# uniform gini a priori   
u = rep(3,  p)#rep(100,p)#
v = rep(0.25, p)#rep(1,p)#

## initial values
k.init = 5
M.na.init = 10
set.seed(222)
C.init = sample(1:k.init,n,replace=T)
cent.init = matrix(sample(1:m[1],k.init+M.na.init,replace=T),nrow = (k.init+M.na.init), ncol=p)
sigma.init = matrix(0.5,nrow = (k.init+M.na.init), ncol=p)
 

### RUN GIBBS SAMPLER ###
set.seed(222)
sim_mvad = gibbs_mix_con(G=30000,
                        burnin = 5000,
                        data=mvad.data,
                        u=u,v=v,
                        Lambda = Lambda,
                        gam = gam,
                        C.init = C.init,
                        k.init = k.init,
                        cent.init = cent.init,
                        sigma.init = sigma.init,
                        M.na.init =M.na.init,
                        M.max=100)

# posterior K
post_k = table(sim_mvad$k[2:30002])/length(2:30002)

# posterior similarity matrix
psm = comp.psm(sim_mvad$C[2:30002,])

# estimated clusters
pred_VI = minVI(psm)$cl
table(pred_VI)


# Figure S16a
xl=15
x11()
par(mar=c(3.5,2,1,1),mgp=c(2,1,0))
plot(post_k,lwd = 2,
     xlab = "k", ylab="", xlim=c(1,xl),axes=F)
segments(1:xl,rep(0,xl),1:xl,prior,col="red",pch=4)
axis(1,1:xl,1:xl,cex.axis=1)
axis(2)
legend("topleft",legend=c("P(K = k)","P(K = k | data)"),
       col=c("red",1),lwd=c(1,2))


# Figure S16b
x11()
par(mar=c(2.5,2.5,1,1),mgp=c(2,1,0))
myplotpsm(psm,classes=pred_VI,ax=F,ay=F)



###### parameters estimation #########
source("../code/gibbs_param_estim2.R")
Kest = length(unique(pred_VI))
cent.last = sigma.last = matrix(NA, Kest, p)
for(i in 1:Kest){
  cent.last[i,] = sim_mvad$Cent[[i]][nrow(sim_mvad$Cent[[i]]),]
  sigma.last[i,] = sim_mvad$Sigma[[i]][nrow(sim_mvad$Cent[[i]]),]
}

set.seed(5)
gibbs_param = gibbs_param_estim(G=5000,
                                gam=gam,
                                data=mvad.data,
                                u=u,
                                v=v,
                                estim_Truth = pred_VI,
                                cent.init = cent.last,
                                sigma.init = sigma.last,
                                Sm.init = sim_mvad$Sm[[length(sim_mvad$Sm)]][1:Kest]
)

### marginal posterior mode of c_jk
post.cent = apply(gibbs_param$Cent,c(2,3),moda)

# figure S17
x11()
par(mar=c(3.5,3.5,1,1),mgp=c(2,1,0))
seqplot(seqdef(post.cent[1:12,],labels=mvad.LAB),type="I",
        xlab="",ylab="clusters",xtlab = months,
        yaxis="all",xaxis="all",cex.axis=0.7)


##########################################################################
### MEDseq analysis ######################################################
##########################################################################
mvad$Location <- factor(apply(mvad[,5L:9L], 1L, function(x) which(x == "yes")), 
                        labels = colnames(mvad[,5L:9L]))
mvad          <- list(covariates = mvad[c(3L:4L, 10L:14L, 87L)], 
                      sequences = mvad[,15L:86L], weights = mvad[,2L])
mvad.cov      <- mvad$covariates
mvad.seq      <- seqdef(mvad$sequences[-c(1L, 2L)],
                        states = c("EM", "FE", "HE", "JL", "SC", "TR"),
                        labels = c("Employment", "Further Education", "Higher Education", 
                                   "Joblessness", "School", "Training"))

mod1 <- MEDseq_fit(mvad.seq, G=11, modtype="UUN", weights=mvad$weights, gating= ~ gcse5eq, 
                   covars=mvad.cov, control=MEDseq_control(noise.gate=FALSE))

x11()
plot(mod1, type="central")
plot(mod1, type="dbsvals")
plot(mod1, type="clusters")
seqplot(seqdef(mod1$params$theta[1:10,],labels=mvad.LAB),type="I",
        xlab="",ylab="clusters",xtlab = months,
        yaxis="all",xaxis="all",cex.axis=0.7)

### siholuette index
library(cluster)
dist_mat = matrix(NA,nrow=nrow(mvad.data),ncol = nrow(mvad.data))

for (i in 1:nrow(mvad.data)){
  for (j in 1:nrow(mvad.data)){
    dist_mat[i,j] = hamming_distance(mvad.data[i,],mvad.data[j,])
  }
}

# Figure S18
x11()
par(mfrow=c(1,2))
sil_vi = silhouette(pred_VI,dmatrix = dist_mat)
plot(sil_vi, main='HMM')
mean(sil_vi[,3])
sd(sil_vi[,3])

sil_medsq = silhouette(mod1$MAP+1,dmatrix = dist_mat)
plot(sil_medsq,main='MEDseq')
mean(sil_medsq[,3])
sd(sil_medsq[,3])
