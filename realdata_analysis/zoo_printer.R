setwd("~/Documents/Split_and_merge_Gibbs_sampling/realdata_analysis")
library(AntMAN)
library(mcclust.ext)
library(ggplot2)
library(tidyverse)
library(pheatmap)
library(LaplacesDemon)
#library(vcd)
rm(list=ls())
source("../code/old_code/complement_functions.R")

zoo=read.table("../data/zoo.data", h=F, sep=",")
nam = zoo$V1
groundTruth = zoo$V18

zoo = as.matrix(zoo[,-c(1,18)]+1)
zoo[,13] = ifelse(zoo[,13]==3,2,
                  ifelse(zoo[,13]==5,3,
                         ifelse(zoo[,13]==6,4,
                                ifelse(zoo[,13]==7,5,
                                       ifelse(zoo[,13]==9,6,
                                              1)))))
n = nrow(zoo)
p = ncol(zoo)

# Mosaic plot
classes = factor(groundTruth,labels=c("mammals", "birds", "reptiles", "fish", 
                                      "amphibians", "insects", "mollusks"))
names(groundTruth) <- classes

zoo_subset <- data.frame(ifelse(zoo[,c(2,5,7)] == 1, "no", "yes"))
colnames(zoo_subset) <- c("Feather","Airborne","Predator")

# Convert to factor if needed
zoo_subset$Feather <- as.factor(zoo_subset$Feather)
zoo_subset$Airborne <- as.factor(zoo_subset$Airborne)
zoo_subset$Predator <- as.factor(zoo_subset$Predator)

# Create a contingency table
zoo_table <- table(zoo_subset$Feather, zoo_subset$Airborne, zoo_subset$Predator)
dimnames(zoo_table) <- list(Feather = levels(zoo_subset$Feather), 
                            Airborne = levels(zoo_subset$Airborne),
                            Predator = levels(zoo_subset$Predator))

# Generate the mosaic plot
#mosaic(zoo_table, shade = T, legend = F)

##### Parameter for chain 

mm = apply(zoo, 2, function(x){length(table(x))})
v = c(rep(6,12), 3, rep(6,3))
w = c(rep(0.25,12), 0.5, rep(0.25,3))

n8 <- TRUE
sam <- FALSE

result_name_base = "Test"
if(n8){
  result_name_base = paste(result_name_base, "Neal8", sep = "_")
}

if(sam){
  result_name_base = paste(result_name_base, "SplitMerge", sep = "_")
}

L_plurale <- c(101, 20, 0, 1, 5) # 5 siccome è log(n)
iterations <- 15000
burnin <- 10000
m <- 3
#t_s <- c(5, 10, 15, 20, 30)
#r_s <- c(5, 10, 15, 20, 30)

t_s <- c(10)
r_s <- c(10)
L_plurale <- c(1)

# Generate all combinations and filter for matches
combinations <- expand.grid(t = t_s, r = r_s)

# Convert to desired format (list of vectors)
sam_params <- split(combinations, seq(nrow(combinations)))
sam_params <- lapply(sam_params, function(x) c(x$t, x$r))

if(n8 == TRUE && sam == TRUE){
  steps <- list(c(1, 1), c(2, 1), c(1, 2))
} else{
  steps <- list(c(1, 1))
}

verbose <- 0
thinning <- 1
# [IMPORTANT] Gamma test improve loglikelihood and acf
gamma <- 0.68

Rcpp::sourceCpp("../code/neal8.cpp")

for(step in steps){
  n8_step <- step[1]
  sam_step <- step[2]

  for(param in sam_params){
    r <- param[1]
    t <- param[2]

    for(l in L_plurale){
      print(paste("Running for L = ", l, " r = ", r, " t = ", t, " n8_step = ", n8_step, " s&m_step = ", sam_step, sep = ""))

      temp_time <- format(Sys.time(), "%Y%m%d_%H%M%S")
      #result_name = "result_"
      result_name <- result_name_base
      # Clean output file
      if(file.exists("output.txt")) {
        # Remove the file
        file.remove("output.txt")
      }
      sink("output.txt")
      if(l == 1){
        results <- run_markov_chain(
                                    gt = groundTruth,
                                    data = zoo, 
                                    attrisize = mm, 
                                    gamma = gamma, 
                                    v = v, 
                                    w = w, 
                                    verbose = verbose, 
                                    m = m, 
                                    iterations = iterations,
                                    c_i = rep(0,nrow(zoo)),
                                    burnin = burnin,
                                    t = t, 
                                    r = r,
                                    neal8 = n8,
                                    split_merge = sam,
                                    n8_step_size = n8_step,
                                    sam_step_size = sam_step, 
                                    thinning = thinning) 
      }
      else if(l == 101){
        results <- run_markov_chain(gt = groundTruth,
                                    data = zoo, 
                                    attrisize = mm, 
                                    gamma = gamma, 
                                    v = v, 
                                    w = w, 
                                    verbose = verbose, 
                                    m = m, 
                                    iterations = iterations,
                                    c_i = seq(1,nrow(zoo)),
                                    burnin = burnin,
                                    t = t, 
                                    r = r,
                                    neal8 = n8,
                                    split_merge = sam,
                                    n8_step_size = n8_step,
                                    sam_step_size = sam_step, 
                                    thinning = thinning)
      }
      else if(l == 0){
        results <- run_markov_chain(gt = groundTruth,
                                    data = zoo, 
                                    attrisize = mm, 
                                    gamma = gamma, 
                                    v = v, 
                                    w = w, 
                                    verbose = verbose, 
                                    m = m, 
                                    iterations = iterations,
                                    c_i = unlist(groundTruth), 
                                    burnin = burnin,
                                    t = t, 
                                    r = r,
                                    neal8 = n8,
                                    split_merge = sam,
                                    n8_step_size = n8_step,
                                    sam_step_size = sam_step, 
                                    thinning = thinning)
      }
      else{
        results <- run_markov_chain(data = zoo, 
                                    attrisize = mm, 
                                    gamma = gamma, 
                                    v = v, 
                                    w = w, 
                                    verbose = verbose, 
                                    m = m, 
                                    iterations = iterations,
                                    L = l,
                                    burnin = burnin,
                                    t = t, 
                                    r = r,
                                    neal8 = n8,
                                    split_merge = sam,
                                    n8_step_size = n8_step,
                                    sam_step_size = sam_step, 
                                    thinning = thinning)
      }
      sink(NULL)
      result_name = paste(result_name, "L", l, sep = "_")
      if(n8){
        result_name = paste(result_name, "M", m, "N8step", n8_step, sep = "_")
      }
      if(sam){
        result_name = paste(result_name, "t", t, sep = "_")
        result_name = paste(result_name, "r", r, sep = "_")
        result_name = paste(result_name, "SM_step", sam_step, sep = "_")
      }
      
      result_name = paste(result_name, "BI", burnin, sep = "_")
      result_name = paste(result_name, "IT", iterations, sep = "_")
      result_name = paste(result_name, "time", results$time, sep = "_")
      
      # Save results
      #filename <- paste("../results/", result_name, l, "_",m, "_", iterations,"_",temp_time, "_S&M",".RData", sep = "")
      filename <- paste("../results/", result_name, ".RData", sep = "")
      save(results, file = filename)
      print(paste("Results for L = ", l, " saved in ", filename, sep = ""))
    }
  }
}

#########
# Posterior metrics
#########
results_dir <- file.path(getwd(), "../results")
dir.exists(results_dir)
print(normalizePath(results_dir))
rdata_files <- list.files(results_dir, full.names = TRUE)

extract_mcmc_parameters <- function(rdata_files, groundTruth = NULL) {
  results_list <- lapply(rdata_files, function(file) {
    # Load results
    load(file)
    
    # Extract parameters from filename
    filename <- basename(file)
    filename_parts <- strsplit(filename, "_")[[1]]
    
    # Verbose parameter extraction
    safe_extract <- function(pattern) {
      # Look for exact pattern match followed by a number
      matches <- which(grepl(paste0("^", pattern, "$"), filename_parts))
      
      if (length(matches) > 0) {
        # Try to extract the next element as value
        value <- tryCatch(
          as.numeric(filename_parts[matches + 1]),
          warning = function(w) NA_real_,
          error = function(e) NA_real_
        )
        
        if (length(value) > 0 && !is.na(value)) {
          return(value)
        }
      }
      
      return(NA_real_)
    }
    
    # Compute metrics
    mcmc_list <- list(ncls = unlist(results$total_cls), logl = results$loglikelihood)
    mcmc_matrix <- do.call(cbind, mcmc_list)
        
    # Calculate ESS and IAT
    ess <- ESS(mcmc_matrix)
    iat <- IAT(mcmc_matrix)
    
    # Compute ARI if groundTruth is provided
    ari <- NA_real_
    if (!is.null(groundTruth)) {
      C <- matrix(unlist(lapply(results$c_i, function(x) x + 1)), 
                  nrow = length(results$total_cls), 
                  ncol = length(groundTruth), 
                  byrow = TRUE)
      VI <- minVI(comp.psm(C))
      ari <- arandi(VI$cl, groundTruth)
    }
    
    # Create data frame
    data.frame(
      filename = filename,
      L = safe_extract("L"),
      M = safe_extract("M"),
      r = safe_extract("r"),
      t = safe_extract("t"),
      burnin = safe_extract("BI"),
      iterations = safe_extract("IT"),
      ESS_ncls = ess["ncls"],
      ESS_logl = ess["logl"],
      IAT = iat,
      ARI = ari,
      time = results$time,
      stringsAsFactors = FALSE
    )
  })
  
  # Combine results, suppressing row names
  do.call(rbind, results_list)
}

# Extract MCMC parameters - IAC, ESS, ARI and time
mcmc_params <- extract_mcmc_parameters(rdata_files, groundTruth)
# così ci sarà da fare solo i grafici e non ricalcolare i parametri e anche plottare grafici r/tempo e t/tempo viene più comodo
#save(mcmc_params, file = "../results/mcmc_params.RData") 
print(mcmc_params)

dev.off()
graphics.off()
# Plot 
idx <- 0
for (file in rdata_files) {
  idx <- idx + 1
  # Print file name 
  print(file)
  l <- L_plurale[idx]
  load(file)
  
  # Extract file name without extension
  file_base <- tools::file_path_sans_ext(basename(file))
  
  # Create a folder for saving plots if it doesn't exist
  output_dir <- paste("../print/plot",file_base, sep = "_")  # Change this to your desired folder
  if (!dir.exists(output_dir)) {
    dir.create(output_dir)
  
  ### First plot - Posterior distribution of the number of clusters
  # Calculation
  post_total_cls = table(unlist(results$total_cls))/length(unlist(results$total_cls))
  #title <- paste("Posterior distribution of the number of clusters ( L =", l, ")")
  df <- data.frame(cluster_found = as.numeric(names(post_total_cls)),
                   rel_freq = as.numeric(post_total_cls))
  # Create plot
  p1 <- ggplot(data = df, aes(x = factor(cluster_found), y = rel_freq)) + 
    geom_col() + 
    labs(
      x = "Cluster Found",
      y = "Relative Frequency",
      #title = title
    ) +
    theme_minimal() +
    scale_x_discrete(drop = FALSE)  # Ensures all cluster_found values are shown
  print(p1)
  ggsave(filename = file.path(output_dir, paste0(substr(file_base,31,60), "_posterior_distribution.png")), plot = p1, bg = "white")
  
  ### Second plot - Trace of number of clusters
  total_cls_df <- data.frame(
    Iteration = seq_along(results$total_cls),
    NumClusters = unlist(results$total_cls)
  )
  
  total_cls_df_long <- total_cls_df %>%
    pivot_longer(cols = starts_with("NumClusters"), names_to = "variable", values_to = "value")
  
  p2 <- ggplot(total_cls_df_long, aes(x = Iteration, y = value)) +
    geom_line() +
    labs(
      x = "Iteration", 
      y = "Number of clusters", 
      #title = paste("Trace of Number of Clusters starting from L =", l)
    ) +
    theme_minimal()
  print(p2)
  ggsave(filename = file.path(output_dir, paste0(substr(file_base,31,60), "_trace_num_clusters.png")), plot = p2, bg = "white")
  
  ### inter plot - Plot the log-likelihood before S&M
  log_likelihood_df_bis <- data.frame(
    Iteration = seq_along(results$loglikelihood),
    LogLikelihood = results$loglikelihood
  )
  
  p3_bis <- ggplot(log_likelihood_df_bis, aes(x = Iteration, y = LogLikelihood)) +
    geom_line() +
    labs(
      x = "Iteration",
      y = "Log-Likelihood",
      #title = "Log-Likelihood Trace"
    ) +
    theme_minimal()
  print(p3_bis)
  ggsave(filename = file.path(output_dir, paste0(substr(file_base,31,60), "_log_likelihood_bfsm.png")), plot = p3_bis, bg = "white")
  
  ### Fourth plot - Posterior similarity matrix
  # Vectorized approach to create the matrix
  C <- matrix(unlist(lapply(results$c_i, function(x) x + 1)), 
              nrow = iterations, 
              ncol = nrow(zoo), 
              byrow = TRUE)
  
  required_packages <- c("spam", "fields", "viridisLite","RColorBrewer","pheatmap")
  for (pkg in required_packages) {
    if (!require(pkg, character.only = TRUE)) {
      install.packages(pkg)
      library(pkg, character.only = TRUE)
    }
  }
  
  psm = comp.psm(C)
  ## estimated clustering
  VI = minVI(psm)
  
  # More informative output
  cat("Cluster Sizes:\n")
  print(table(VI$cl))
  
  cat("\nAdjusted Rand Index:", arandi(VI$cl, groundTruth), "\n")
  arandi(VI$cl, groundTruth)
  png(filename = file.path(output_dir, paste0(file_base, "matrix.png")), width = 800, height = 800)
  myplotpsm(psm, classes=VI$cl, ax=F, ay=F)
  dev.off()  # Close the device to save the first plot

  # Fifth plot - Auto-correlation plot
  mcmc_list <- list( ncls = unlist(results$total_cls), logl = results$loglikelihood)
  mcmc_matrix <- do.call(cbind, mcmc_list)
  png(filename = file.path(output_dir, paste0(substr(file_base,31,60), "acf.png")), width = 800, height = 800)
  acf(mcmc_matrix)
  dev.off()
  #graphics.off()
  # Save the second plot
  # png(filename = file.path(output_dir, paste0(substr(file_base,31,60), "m_gt.png")), 
  #     width = 800, height = 800)
  # myplotpsm_gt(psm, groundTruth, classes=VI$cl, ax=F, ay=F)
  # dev.off()  # Close the device to save the second plot
  # graphics.off()
  # png(filename = file.path(output_dir, paste0(substr(file_base,31,60), "m_s.png")), 
  #     width = 800, height = 800)
  # myplotpsm_gt_sep(psm, groundTruth, classes=VI$cl, gt = 1, ax=F, ay=F)
  # dev.off()

  # graphics.off()
  }
}

# #=========================================================================================
# # Gibbs sampler HMM
# #=========================================================================================
# source('../code/gibbs_sampler.R', echo=TRUE)

# Kstar  = 7
# Lambda = 7
# gam    = AntMAN::AM_find_gamma_Pois(n=nrow(zoo),Lambda=Lambda,Kstar=Kstar)
# prior = AM_prior_K_Pois(n=nrow(zoo), gam, Lambda = Lambda)

# u = c(rep(6,12),3,rep(6,3))
# v = c(rep(0.25,12),0.5,rep(0.25,3))

# set.seed(57)
# sim_zoo = gibbs_mix_con(G=25000,
#                         burnin = 5000,
#                         data=zoo,
#                         u=u,v=v,
#                         Lambda = Lambda,
#                         gam = gam)

# filename <- paste("../results/", 'sim_zoo', ".RData", sep = "")
# save(sim_zoo, file = filename)

# # posterior K
# post_k = table(sim_zoo$k[2:25002])/length(2:25002)


# # Figure S2a
# xl=15
# x11()
# par(mar=c(3.5,2,1,1),mgp=c(2,1,0))
# plot(post_k,lwd = 2,
#      xlab = "k", ylab="", xlim=c(1,xl),axes=F)
# segments(1:xl,rep(0,xl),1:xl,prior,col="red",pch=4)
# axis(1,1:xl,1:xl,cex.axis=1)
# axis(2)
# legend("topleft",legend=c("P(K = k)","P(K = k | data)"),
#        col=c("red",1),lwd=c(1,2))


# ## posterior similarity matrix
# psm = comp.psm(sim_zoo$C[2:25002,])

# ## estimated clustering
# VI = minVI(psm)
# table(VI$cl)
# arandi(VI$cl,groundTruth)


# # Figure 2b
# x11()
# par(mar=c(2.5,2.5,1,1),mgp=c(2,1,0))
# myplotpsm(psm,classes=VI$cl,ax=F,ay=F)
# myplotpsm_gt_sep(psm,groundTruth,classes=VI$cl,ax=F,ay=F)



# #=========================================================================================
# # Gibbs sampler HMM common sigma
# #=========================================================================================
# source('../code/gibbs_sampler_common_sigma.R',echo=T)

# Kstar  = 7
# Lambda = 7
# gam    = AntMAN::AM_find_gamma_Pois(n=nrow(zoo),Lambda=Lambda,Kstar=Kstar)
# prior = AM_prior_K_Pois(n=nrow(zoo), gam, Lambda = Lambda)

# u = c(rep(6,12),3,rep(6,3))
# v = c(rep(0.25,12),0.5,rep(0.25,3))

# set.seed(10091995)
# sim_zoo2 = gibbs_ham(G = 10000,
#                      burnin = 2000,
#                      thin = 1,
#                      data = zoo,
#                      eta = c(rep(0.2,30)),
#                      gam =  gam,
#                      Lambda = Lambda,
#                      M.init = 10,
#                      a=1,
#                      b=0.01)

# ## posterior similarity matrix
# psm2 = comp.psm(sim_zoo2$C)


# ## estimated clustering
# VI2= minVI(psm2)
# table(VI2$cl)
# arandi(VI2$cl,groundTruth) 

# #=========================================================================================
# # competitors
# #=========================================================================================
# source('../code/competitors_functions.R')


# ############## HD-vector #################
# HD_rand = NULL
# set.seed(1185)
# for(i in 1:100){
#   HD_output = CategorialCluster(zoo)[[1]]
#   HD_rand[i] = arandi(HD_output,groundTruth)
# }

# mean(HD_rand)
# sd(HD_rand)


# ############## K-modes #################
# library(klaR)
# k_mod_rand7 =  NULL
# set.seed(18)
# for(i in 1:100){
#   kmodes_cluster7 = kmodes(zoo,7)$cluster
  
#   # aRand index
#   k_mod_rand7[i] = arandi(kmodes_cluster7,groundTruth)
# }

# mean(k_mod_rand7)
# sd(k_mod_rand7)



# ####### silhoutte index
# library(cluster)
# dist_mat = matrix(NA,nrow=nrow(zoo),ncol = nrow(zoo))

# for (i in 1:nrow(zoo)){
#   for (j in 1:nrow(zoo)){
#     dist_mat[i,j] = hamming_distance(zoo[i,],zoo[j,])
#   }
# }

# x11()
# par(mfrow=c(1,2))
# sil_vi = silhouette(VI$cl,dmatrix = dist_mat)
# plot(sil_vi, main='HMM')

# sil_hd = silhouette(HD_output,dmatrix = dist_mat)
# plot(sil_hd,main='HD')

# summary(sil_vi)
# mean(sil_vi[,3])
# var(sil_vi[,3])
# summary(sil_vi[,3])

# summary(sil_hd)
# mean(sil_hd[,3])
# var(sil_hd[,3])
# summary(sil_hd[,3])
