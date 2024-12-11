setwd("~/Documents/Split_and_merge_Gibbs_sampling/realdata_analysis")
library(AntMAN)
library(mcclust.ext)
library(ggplot2)
library(tidyverse)
source("../code/complement_functions.R")

#=========================================================================================
# Loading data
#=========================================================================================

zoo=read.table("../data/zoo.data", h=F, sep=",")
nam = zoo$V1
groundTruth = zoo$V18
#classes = factor(groundTruth,labels=c("mammals", "birds", "reptiles", "fish", 
                                      #"amphibians", "insects", "mollusks"))
#names(groundTruth)<-classes

#=========================================================================================
# Data cleaning
#=========================================================================================

zoo = as.matrix(zoo[,-c(1,18)]+1)

zoo[,13] = ifelse(zoo[,13]==3,2,
                  ifelse(zoo[,13]==5,3,
                         ifelse(zoo[,13]==6,4,
                                ifelse(zoo[,13]==7,5,
                                       ifelse(zoo[,13]==9,6,
                                              1)))))

n = nrow(zoo)
p = ncol(zoo)
mm = apply(zoo, 2, function(x){length(table(x))})

#=========================================================================================
# Neal sampler
#=========================================================================================

## Obbligatorie per Filippo perché RcppGSL non trova le librerie GSL e nemmeno RcppGSL.h
# First, use the explicit include path from RcppGSL
Sys.setenv("PKG_CXXFLAGS" = paste0("-I/home/filippo/R/x86_64-pc-linux-gnu-library/4.4/RcppGSL/include", 
                                   " -I/usr/local/include"))

# Include full library paths and libraries
Sys.setenv("PKG_LIBS" = "-L/usr/local/lib -lgsl -lgslcblas -lm")

u = c(rep(6,12),3,rep(6,3))
v = c(rep(0.25,12),0.5,rep(0.25,3))

Rcpp::sourceCpp("../code/test.cpp")

L_plurale <- c(7)
initial_assignment_bool <- c(TRUE, FALSE)
iterations <- 25000
burnin <- 8000
m <- 3
# Create 3 plot with different starting point and or not initial assignment
for(init_ass_bool in initial_assignment_bool){
  for(l in L_plurale){
    temp_time <- format(Sys.time(), "%Y%m%d_%H%M%S")
    result_name = "result_"
    if(init_ass_bool){
      results <- run_markov_chain(data = zoo, 
                                  attrisize = mm, 
                                  gamma = 0.68, 
                                  v = v, 
                                  w = u, 
                                  verbose = 0, 
                                  m = m, 
                                  iterations = iterations, 
                                  L = l, 
                                  c_i = unlist(groundTruth), 
                                  burnin = burnin)
      result_name = paste(result_name, "init_ass_", sep="")
    }
    else{
      results <- run_markov_chain(data = zoo, 
                                  attrisize = mm, 
                                  gamma = 0.68, 
                                  v = v, 
                                  w = u, 
                                  verbose = 0, 
                                  m = m, 
                                  iterations = iterations, 
                                  L = l, 
                                  burnin = burnin)
    }
    # Save results
    filename <- paste("../results/", result_name, l, "_",m, "_", iterations,"_",temp_time,".RData", sep = "")
    save(results, file = filename)
    print(paste("Results for L = ", l, " saved in ", filename, sep = ""))

    ### First plot - Posterior distribution of the number of clusters
    # Calculation
    post_total_cls = table(unlist(results$total_cls))/length(unlist(results$total_cls))
    title <- paste("Posterior distribution of the number of clusters ( L =", l, ")")
    df <- data.frame(cluster_found = as.numeric(names(post_total_cls)),
                    rel_freq = as.numeric(post_total_cls))
    # Create plot
    p <-ggplot(data = df, aes(x = factor(cluster_found), y = rel_freq)) + 
      geom_col() + 
      labs(
        x = "Cluster Found",
        y = "Relative Frequency",
        title = title
      ) +
      theme_minimal() +
      scale_x_discrete(drop = FALSE)  # Ensures all cluster_found values are shown
    # Save plot
    save_name = "post_total_cls_"
    if(init_ass_bool){
      save_name = paste(save_name, "init_assignment_", sep="")
    }
    filename <- paste("../plot/", save_name , l, "_", temp_time,".png", sep = "")
    ggsave(filename, plot = p)

    ### Second plot - Trace of number of clusters
    total_cls_df <- data.frame(
      Iteration = seq_along(results$total_cls),
      NumClusters = unlist(results$total_cls)
    )

    total_cls_df_long <- total_cls_df %>%
    pivot_longer(cols = starts_with("NumClusters"), names_to = "variable", values_to = "value")

    p <- ggplot(total_cls_df_long, aes(x = Iteration, y = value)) +
      geom_line() +
      labs(
        x = "Iteration", 
        y = "Number of clusters", 
        title = paste("Trace of Number of Clusters starting from L =", l)
      ) +
      theme_minimal()

    # Save plot
    save_name = "trace_cls_starting_point_"
    if(init_ass_bool){
      save_name = paste(save_name, "init_assignment_", sep="")
    }
    filename <- paste("../plot/", save_name, l, "_", temp_time, ".png", sep = "")
    ggsave(filename, plot = p)

    ### Third plot - Plot the log-likelihood
    log_likelihood_df <- data.frame(
      Iteration = seq_along(results$loglikelihood),
      LogLikelihood = results$loglikelihood
    )

    p <- ggplot(log_likelihood_df, aes(x = Iteration, y = LogLikelihood)) +
      geom_line() +
      labs(
        x = "Iteration",
        y = "Log-Likelihood",
        title = "Log-Likelihood Trace"
      ) +
      theme_minimal()

    # Save plot
    save_name = "loglikelihood_"
    if(init_ass_bool){
      save_name = paste(save_name, "init_assignment_", sep="")
    }
    filename <- paste("../plot/", save_name, l, "_", temp_time, ".png", sep = "")
    ggsave(filename, plot = p)

  }
}


### Trace of c_i history for specific observation
choosen_idx <- 100
temp <- sapply(seq_along(results$c_i), function(i) {
  unlist(results$c_i[[i]])[choosen_idx]
})

c_i_df <- data.frame(
  Iteration = seq_along(temp),
  ClusterAssignment = temp
)

ggplot(c_i_df, aes(x = Iteration, y = ClusterAssignment)) +
  geom_line() +
  labs(
    x = "Iteration", 
    y = "Cluster Assignment", 
    title = paste("Trace of c_i History for Observation ", choosen_idx)
  ) +
  theme_minimal()

### Posterior similarity matrix
# Create matrix from c_i 
C <- matrix(NA, nrow = iterations, ncol = nrow(zoo))

for(i in 1:iterations){
  C[i, ] <- unlist(results$c_i[i]) + 1
}

psm = comp.psm(C)
## estimated clustering
VI = minVI(psm)
table(VI$cl) 
arandi(VI$cl, groundTruth)
myplotpsm(psm, classes=VI$cl, ax=F, ay=F)

#=========================================================================================
# Gibbs sampler HMM
#=========================================================================================
source('../code/gibbs_sampler.R', echo=TRUE)

Kstar  = 7
Lambda = 7
gam    = AntMAN::AM_find_gamma_Pois(n=nrow(zoo),Lambda=Lambda,Kstar=Kstar)
prior = AM_prior_K_Pois(n=nrow(zoo), gam, Lambda = Lambda)

u = c(rep(6,12),3,rep(6,3))
v = c(rep(0.25,12),0.5,rep(0.25,3))

set.seed(57)
sim_zoo = gibbs_mix_con(G=25000,
                        burnin = 5000,
                        data=zoo,
                        u=u,v=v,
                        Lambda = Lambda,
                        gam = gam)

# posterior K
post_k = table(sim_zoo$k[2:25002])/length(2:25002)


# Figure S2a
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


## posterior similarity matrix
psm = comp.psm(sim_zoo$C[2:25002,])

## estimated clustering
VI = minVI(psm)
table(VI$cl)
arandi(VI$cl,groundTruth)


# Figure 2b
x11()
par(mar=c(2.5,2.5,1,1),mgp=c(2,1,0))
myplotpsm(psm,classes=VI$cl,ax=F,ay=F)

#=========================================================================================
# Gibbs sampler HMM common sigma
#=========================================================================================
source('../code/gibbs_sampler_common_sigma.R',echo=T)

Kstar  = 7
Lambda = 7
gam    = AntMAN::AM_find_gamma_Pois(n=nrow(zoo),Lambda=Lambda,Kstar=Kstar)
prior = AM_prior_K_Pois(n=nrow(zoo), gam, Lambda = Lambda)

u = c(rep(6,12),3,rep(6,3))
v = c(rep(0.25,12),0.5,rep(0.25,3))

set.seed(10091995)
sim_zoo2 = gibbs_ham(G = 10000,
                     burnin = 2000,
                     thin = 1,
                     data = zoo,
                     eta = c(rep(0.2,30)),
                     gam =  gam,
                     Lambda = Lambda,
                     M.init = 10,
                     a=1,
                     b=0.01)

## posterior similarity matrix
psm2 = comp.psm(sim_zoo2$C)


## estimated clustering
VI2= minVI(psm2)
table(VI2$cl)
arandi(VI2$cl,groundTruth) 

#=========================================================================================
# competitors
#=========================================================================================
source('../code/competitors_functions.R')


############## HD-vector #################
HD_rand = NULL
set.seed(1185)
for(i in 1:100){
  HD_output = CategorialCluster(zoo)[[1]]
  HD_rand[i] = arandi(HD_output,groundTruth)
}

mean(HD_rand)
sd(HD_rand)


############## K-modes #################
library(klaR)
k_mod_rand7 =  NULL
set.seed(18)
for(i in 1:100){
  kmodes_cluster7 = kmodes(zoo,7)$cluster

  # aRand index
  k_mod_rand7[i] = arandi(kmodes_cluster7,groundTruth)
}

mean(k_mod_rand7)
sd(k_mod_rand7)



####### silhoutte index
library(cluster)
dist_mat = matrix(NA,nrow=nrow(zoo),ncol = nrow(zoo))

for (i in 1:nrow(zoo)){
  for (j in 1:nrow(zoo)){
    dist_mat[i,j] = hamming_distance(zoo[i,],zoo[j,])
  }
}

x11()
par(mfrow=c(1,2))
sil_vi = silhouette(VI$cl,dmatrix = dist_mat)
plot(sil_vi, main='HMM')

sil_hd = silhouette(HD_output,dmatrix = dist_mat)
plot(sil_hd,main='HD')

summary(sil_vi)
mean(sil_vi[,3])
var(sil_vi[,3])
summary(sil_vi[,3])

summary(sil_hd)
mean(sil_hd[,3])
var(sil_hd[,3])
summary(sil_hd[,3])
