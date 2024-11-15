library(MBCbook)
library(AntMAN)
library(mcclust.ext)

source('../code/gibbs_sampler.R', echo=TRUE)
source('../code/complement_functions.R')

#=========================================================================================
# Loading data
#=========================================================================================

data('usps358')
X   = as.matrix(usps358[,-1])
cls = usps358[,1]

#=========================================================================================
# Data cleaning
#=========================================================================================

breaks  = c(-0.001,0.001,0.75,1,1.25,1.5,2)
m       = length(breaks)-1
classes = paste(breaks[1:m],(breaks[2:(m+1)]),sep="--|")
dati.m  = apply(X,2,cut,breaks = breaks,labels = classes)

#Dropping columns with less than m attributes
nclas<- rep(0,256)
for(i in 1:256){
  nclas[i]= length(table(dati.m[,i]))
}

for(i in 1:length(classes)){
  dati.m[dati.m==classes[i]]=i
}
dati.m = apply(dati.m, 2,as.numeric)
to_omit=which(nclas<m)
data.clean=as.data.frame(dati.m[,-to_omit])


# Figure 3
digit = c(1696,1354,681,482,206,1629,139,609,699,281)
col_digit = colorRampPalette(c("white","black"))
x11()
par(mfrow=c(2,5))
for(i in digit){
  image(t(matrix(dati.m[i,],ncol =16, byrow=TRUE)[16:1 ,]),
        col = col_digit(6),
        main="",axes=F);box()
}


#=========================================================================================
# Gibbs sampler
#=========================================================================================

#gamma = AntMAN::AM_find_gamma_Pois(n=nrow(data.clean),Lambda = 3,Kstar = 3) 
gamma = 0.1514657
u     = rep(3,ncol(data.clean))
v     = rep(0.5,ncol(data.clean))

set.seed(35)
gibbs = gibbs_mix_con(G=50000,
                      burnin = 10000,
                      data=data.clean,
                      u=u,
                      v=v,
                      Lambda = 3,
                      gam = gamma)


# posterior K
post_k = table(gibbs$k[2:50000])/length(2:50000)

# posterior similarity matrix
psm = comp.psm(gibbs$C[2:50000,])
# estimated clustering
pred_VI = minVI(psm)$cl


# Figure
x11()
par(mar=c(2.5,2.5,1,1),mgp=c(2,1,0))
myplotpsm(psm,classes=pred_VI)



############### parameter estimation given the partition ##############
source("../code/Gibbs_param_estim.R")

Kest = length(unique(pred_VI))
cent.last = sigma.last = matrix(NA, Kest, p)
for(i in 1:Kest){
  cent.last[i,] = gibbs$Cent[[i]][nrow(gibbs$Cent[[i]]),]
  sigma.last[i,] = gibbs$Sigma[[i]][nrow(gibbs$Cent[[i]]),]
}

set.seed(5)
gibbs_param = gibbs_param_estim(G=2000,
                                gam=gamma,
                                data=data.clean,
                                u=u,
                                v=v,
                                estim_Truth = pred_VI,
                                cent.init = cent.last,
                                sigma.init = sigma.last,
                                Sm.init = gibbs$Sm[[length(gibbs$Sm)]][1:Kest]
)

### marginal posterior mode of c_jk
post.cent= matrix(NA,Kest,256)
post.cent[,-to_omit] = apply(gibbs_param$Cent,c(2,3),moda)
post.cent[,to_omit] = cls[to_omit]

# Figure 5
col_digit = colorRampPalette(c("white","black"))
par(mfrow=c(2,4),mar=c(1,1,1,1))
for(i in 1:Kest){
  image(t(matrix(t(post.cent[i,]),ncol =16, byrow=TRUE)[16:1 ,]),
        col= col_digit(6),axes=F); box()
}


### posterior median of sigma_jk
post.sigma = apply(gibbs_param$Sigma, c(2,3), median)


##############################################
############ fitting LCM using Rmixmod  ######

## remove pixels with less than 6 modalities

to_omit=which(nclas<m)
dati.omit=as.data.frame(dati.m[,-to_omit])

for(i in 1:ncol(dati.omit)){
  dati.omit[,i] = as.factor(dati.omit[,i])
}

nb = 2:30
xem <- mixmodCluster(dati.omit, nbCluster = nb, 
                     criterion = c("BIC","ICL","NEC"),
                     model=mixmodMultinomialModel(listModels="Binary_pk_Ekj"),
                     seed = 18)

## select K
nbC = NULL
criteria = matrix(NA,length(nb),3)
colnames(criteria) = xem@criterion
for(i in 1:length(nb)){
  criteria[i,] = xem@results[[i]]@criterionValue
  nbC[i] = xem@results[[i]]@nbCluster
}

criteria = criteria[order(nbC),]

# Figure 4
x11()
par(mfrow=c(1,2),mar=c(3.2,3.2,1,1),mgp=c(2.2,1,0),cex.axis=0.8)
for(j in 1:2){
  plot(nb,criteria[,j],ylab=colnames(criteria)[j],type="b",
       xlab="k",cex.axis=0.8,axes=F)
  axis(1,seq(min(nb),max(nb),1))
  axis(2);  box()
}

