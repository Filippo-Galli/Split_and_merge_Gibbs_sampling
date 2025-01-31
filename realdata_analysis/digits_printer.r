setwd("~/Documents/Split_and_merge_Gibbs_sampling/realdata_analysis")
library(AntMAN)
library(mcclust.ext)
library(MBCbook)
library(ggplot2)
library(tidyverse)
library(pheatmap)
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
data.clean<-data.clean[1:101,1:16]

#gamma = AntMAN::AM_find_gamma_Pois(n=nrow(data.clean),Lambda = 3,Kstar = 3) 
gamma = 0.1514657
v     = rep(3,ncol(data.clean))
w     = rep(0.5,ncol(data.clean))
mm = apply(data.clean, 2, function(x){length(table(x))})

#=========================================================================================
# Neal sampler
#=========================================================================================

## Obbligatorie per Filippo perché RcppGSL non trova le librerie GSL e nemmeno RcppGSL.h
# First, use the explicit include path from RcppGSL
#Sys.setenv("PKG_CXXFLAGS" = paste0("-O3 ", "-I/home/filippo/R/x86_64-pc-linux-gnu-library/4.4/RcppGSL/include", 
#                                   " -I/usr/local/include"))

#Sys.setenv("PKG_CXXFLAGS" = paste0('-I"C:/Users/clau7/AppData/Local/R/win-library/4.4/RcppGSL/include"', " -I/usr/local/include"))
system.file("include", package = "RcppGSL")
Sys.setenv("PKG_CXXFLAGS" = paste(Sys.getenv("PKG_CXXFLAGS"), 
                                  "-I", system.file("include", package = "RcppGSL")))
# Include full library paths and libraries
Sys.setenv("PKG_LIBS" = "-L/usr/local/lib -lgsl -lgslcblas -lm")


Rcpp::sourceCpp("../code/neal8.cpp")
n8 <- TRUE
sam <- FALSE
step <- 1

result_name_base = "TestDIGITS_"
if(n8){
  result_name_base = paste(result_name_base, "Neal8", sep = "_")
}

if(sam){
  result_name_base = paste(result_name_base, "SplitMerge", sep = "_")
}

L_plurale <- c(1)
iterations <- 40000
burnin <- 10000
maux <- 3
t <- 60
r <- 60
#data.clean<-data.clean[1:101,1:16]
data.clean<-as.matrix(data.clean)

for(l in L_plurale){
  temp_time <- format(Sys.time(), "%Y%m%d_%H%M%S")
  #result_name = "result_"
  result_name <- result_name_base
  # Clean output file
  if(file.exists("output.txt")) {
    # Remove the file
    file.remove("output.txt")
  }
  #sink("output.txt")
  if(l == 1){
    results <- run_markov_chain(data = data.clean, 
                                attrisize = mm, 
                                gamma = gamma, 
                                v = v, 
                                w = w, 
                                verbose = 0, 
                                m = maux, 
                                iterations = iterations,
                                c_i = rep(0,nrow(data.clean)),
                                burnin = burnin,
                                t = t, 
                                r = r,
                                neal8 = n8,
                                split_merge = sam,
                                n8_step_size = step)
  }
  else if(l == 101){
    results <- run_markov_chain(data = data.clean, 
                                attrisize = mm, 
                                gamma = gamma, 
                                v = v, 
                                w = w, 
                                verbose = 0, 
                                m = maux, 
                                L=7,
                                iterations = iterations,
                                c_i = seq(1,nrow(data.clean)),
                                burnin = burnin,
                                t = t, 
                                r = r,
                                neal8 = n8,
                                split_merge = sam,
                                n8_step_size = step)
  }
  else if(l == 0){
    results <- run_markov_chain(data = data.clean, 
                                attrisize = mm, 
                                gamma = gamma, 
                                v = v, 
                                w = w, 
                                verbose = 0, 
                                m = maux, 
                                iterations = iterations,
                                c_i = unlist(groundTruth), 
                                burnin = burnin,
                                t = t, 
                                r = r,
                                neal8 = n8,
                                split_merge = sam,
                                n8_step_size = step)
  }
  else{
    results <- run_markov_chain(data = data.clean, 
                                attrisize = mm, 
                                gamma = gamma, 
                                v = v, 
                                w = w, 
                                verbose = 0, 
                                m = maux, 
                                iterations = iterations,
                                L = l,
                                burnin = burnin,
                                t = t, 
                                r = r,
                                neal8 = n8,
                                split_merge = sam,
                                n8_step_size = step)
  }
  #sink()
  #result_name = paste(result_name, "init_ass_", sep="")
  result_name = paste(result_name, "L", l, sep = "_")
  if(n8){
    result_name = paste(result_name, "M", m, sep = "_")
  }
  if(sam){
    result_name = paste(result_name, "t", t, sep = "_")
    result_name = paste(result_name, "r", r, sep = "_")
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


### Posterior similarity matrix
results_dir <- file.path(getwd(), "../results")
dir.exists(results_dir)
print(normalizePath(results_dir))
rdata_files <- list.files(results_dir, full.names = TRUE)

dev.off()  # Close any open graphic devices
graphics.off()  # Close all graphic devices

for (file in rdata_files) {
  # Print file name 
  print(file)
  l <- 7
  load(file)
  
  # Extract file name without extension
  file_base <- tools::file_path_sans_ext(basename(file))
  
  # Create a folder for saving plots if it doesn't exist
  output_dir <- paste("../print/plot",file_base, sep = "_")  # Change this to your desired folder
  if (!dir.exists(output_dir)) {
    dir.create(output_dir)
  }
  
  
  
  ### First plot - Posterior distribution of the number of clusters
  # Calculation
  post_total_cls = table(unlist(results$total_cls))/length(unlist(results$total_cls))
  title <- paste("Posterior distribution of the number of clusters ( L =", l, ")")
  df <- data.frame(cluster_found = as.numeric(names(post_total_cls)),
                   rel_freq = as.numeric(post_total_cls))
  # Create plot
  p1 <- ggplot(data = df, aes(x = factor(cluster_found), y = rel_freq)) + 
    geom_col() + 
    labs(
      x = "Cluster Found",
      y = "Relative Frequency",
      title = title
    ) +
    theme_minimal() +
    scale_x_discrete(drop = FALSE)  # Ensures all cluster_found values are shown
  print(p1)
  ggsave(filename = file.path(output_dir, paste0(file_base, "_posterior_distribution.png")), plot = p1, bg = "white")
  
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
      title = paste("Trace of Number of Clusters starting from L =", l)
    ) +
    theme_minimal()
  print(p2)
  ggsave(filename = file.path(output_dir, paste0(file_base, "_trace_num_clusters.png")), plot = p2, bg = "white")
  
  ### Third plot - Plot the log-likelihood
  log_likelihood_df <- data.frame(
    Iteration = seq_along(results$loglikelihood),
    LogLikelihood = results$loglikelihood
  )
  
  p3 <- ggplot(log_likelihood_df, aes(x = Iteration, y = LogLikelihood)) +
    geom_line() +
    labs(
      x = "Iteration",
      y = "Log-Likelihood",
      title = "Log-Likelihood Trace"
    ) +
    theme_minimal()
  print(p3)
  ggsave(filename = file.path(output_dir, paste0(file_base, "_log_likelihood.png")), plot = p3, bg = "white")
  
  ### inter plot - Plot the log-likelihood before S&M
  log_likelihood_df_bis <- data.frame(
    Iteration = seq_along(results$loglikelihood_bfsam),
    LogLikelihood = results$loglikelihood_bfsam
  )
  
  p3_bis <- ggplot(log_likelihood_df_bis, aes(x = Iteration, y = LogLikelihood)) +
    geom_line() +
    labs(
      x = "Iteration",
      y = "Log-Likelihood",
      title = "Log-Likelihood Trace"
    ) +
    theme_minimal()
  print(p3_bis)
  ggsave(filename = file.path(output_dir, paste0(file_base, "_log_likelihood_bfsm.png")), plot = p3_bis, bg = "white")
  
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
  png(filename = file.path(output_dir, paste0(file_base, "matrix.png")), 
      width = 800, height = 800)
  myplotpsm(psm, classes=VI$cl, ax=F, ay=F)
  dev.off()  # Close the device to save the first plot
  dev.off()
  
  # Save the second plot
  png(filename = file.path(output_dir, paste0(file_base, "m_gt.png")), 
      width = 800, height = 800)
  myplotpsm_gt(psm, groundTruth, classes=VI$cl, ax=F, ay=F)
  dev.off()  # Close the device to save the second plot
  
  png(filename = file.path(output_dir, paste0(file_base, "m_s.png")), 
      width = 800, height = 800)
  myplotpsm_gt_sep(psm, groundTruth, classes=VI$cl, gt = 1, ax=F, ay=F)
  dev.off()
  
  graphics.off()
}

rdata_files[3]
load(rdata_files[3])
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

myplotpsm_gt_lab(psm, groundTruth, classes=VI$cl, ax=F, ay=F)
myplotpsm_gt_sep_lab(psm, groundTruth, classes=VI$cl, gt = 1, ax=F, ay=F)

table(groundTruth)
VI$cl

groundTruth_indices <- which(groundTruth == 3)
VI_indices <- which(VI$cl == 7)

# Compute the symmetric difference (in one but not both)
symmetric_difference <- setdiff(union(groundTruth_indices, VI_indices), 
                                intersect(groundTruth_indices, VI_indices))
intersection_index <- intersect(groundTruth_indices, VI_indices)

nam[symmetric_difference]
nam[VI_indices]
