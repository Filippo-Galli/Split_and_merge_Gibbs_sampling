setwd("~/Documents/test/realdata_analysis")
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

v = c(rep(6,12),3,rep(6,3))
w = c(rep(0.25,12),0.5,rep(0.25,3))

#sink("output.txt")
Rcpp::sourceCpp("../code/launcher.cpp")
#sink(NULL)

burnin <- 5000
iterations <- 1000
m = 3
temp_time <- format(Sys.time(), "%Y%m%d_%H%M%S")
result_name = "result_"
results <- run_markov_chain(data = zoo, 
                     attrisize = mm, 
                     gamma = 0.68, 
                     v = v, 
                     w = w, 
                     verbose = 0, 
                     m = m, 
                     iterations = iterations,
                     c_i = unlist(groundTruth), 
                     burnin = burnin,
                     t = 30, 
                     r = 30,
                     neal8 = TRUE,
                     split_merge = FALSE)
result_name = paste(result_name, "init_ass_", sep="")

# Save results
filename <- paste("../results/", result_name, "_",m, "_", iterations,"_",temp_time,".RData", sep = "")
save(results, file = filename)

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

  ### Fourth plot - Posterior similarity matrix
  # Vectorized approach to create the matrix
  C <- matrix(unlist(lapply(results$c_i, function(x) x + 1)), 
            nrow = iterations, 
            ncol = nrow(zoo), 
            byrow = TRUE)

  required_packages <- c("spam", "fields", "viridisLite")
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
  myplotpsm(psm, classes=VI$cl, ax=F, ay=F)
}

