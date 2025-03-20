#=========================================================================================
# Library loading 
#=========================================================================================
setwd("~/Split_and_merge_Gibbs_sampling/realdata_analysis")
library(AntMAN)
library(mcclust.ext)
library(MBCbook)
library(ggplot2)
library(tidyverse)
library(pheatmap)
source('../code/old_code/gibbs_sampler.R', echo=TRUE)
source('../code/old_code/complement_functions.R')

#=========================================================================================
# Loading data
#=========================================================================================

data('usps358')
X   = as.matrix(usps358[,-1])
gt = usps358[,1]

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

#=========================================================================================
# Gibbs sampler
#=========================================================================================
#gamma = AntMAN::AM_find_gamma_Pois(n=nrow(data.clean),Lambda = 3,Kstar = 3) 
gamma = 0.1514657
v     = rep(3,ncol(data.clean))
w     = rep(0.5,ncol(data.clean))
mm = apply(data.clean, 2, function(x){length(table(x))})

# Info for the chain + thinning
verbose <- 0
thinning <- 1
sink <- FALSE
sink_file <- "output.txt"

# Test to conduct
L_plurale <- c(101, 20, 0, 1, 5)
iterations <- 10000
burnin <- 5000
m <- 3

t_s <- c(5)
r_s <- c(5)
combinations <- expand.grid(t = t_s, r = r_s) # Generate combinations

steps <- list(c(1, 1), c(10, 1)) # first: how much iter should jump n8

# Convert to desired format (list of vectors)
sam_params <- split(combinations, seq(nrow(combinations)))
sam_params <- lapply(sam_params, function(x) c(x$t, x$r))

# Algorithm to use
n8 <- TRUE
sam <- TRUE

## File name base
result_name_base <- "Test"
if (n8) {
  result_name_base <- paste(result_name_base, "Neal8", sep = "_")
}
if (sam) {
  result_name_base <- paste(result_name_base, "SplitMerge", sep = "_")
}

Rcpp::sourceCpp("../code/neal8.cpp")

#=========================================================================================
# Run the chain 
#=========================================================================================

for (step in steps) {
  n8_step <- step[1]
  sam_step <- step[2]
  for (param in sam_params) {
    r <- param[1]
    t <- param[2]
    for (l in L_plurale) {
      print(paste("Testing for L = ", l, " r = ", r, " t = ", t, " n8_step = ", n8_step, " s&m_step = ", sam_step, sep = "")) # nolint: line_length_linter. 
      # file name construction for the output
      temp_time <- format(Sys.time(), "%Y%m%d_%H%M%S")
      result_name <- result_name_base
      result_name <- paste(result_name, "L", l, sep = "_")
      if (n8) {
        result_name <- paste(result_name, "M", m, "N8step", n8_step, sep = "_")
      }
      if (sam) {
        result_name <- paste(result_name, "t", t, sep = "_")
        result_name <- paste(result_name, "r", r, sep = "_")
        result_name <- paste(result_name, "SM_step", sam_step, sep = "_")
      }
      result_name <- paste(result_name, "BI", burnin, sep = "_")
      result_name <- paste(result_name, "IT", iterations, sep = "_")

      # Clean output file
      if (file.exists(sink_file)) {
        file.remove(sink_file)
      }

      # If sink is TRUE, then the output will be saved in the file
      if (sink) {
        sink(sink_file)
      }

      # Initial assignments
      if (l == 1) {
        initial_assignments <- rep(0, nrow(X))
      } else if (l == 0) {
        initial_assignments <- unlist(gt)
      } else if (l == 101) {
        initial_assignments <- seq(1, nrow(X))
      } else {
        initial_assignmenrs <- NULL
      }

      # Running the chain
      results <- run_markov_chain(gt = gt,
                                  data = zoo,
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
                                  c_i = initial_assignments,
                                  r = r,
                                  neal8 = n8,
                                  split_merge = sam,
                                  n8_step_size = n8_step,
                                  sam_step_size = sam_step,
                                  thinning = thinning)

      if (sink) {
        sink(NULL)
      }

      result_name <- paste(result_name, "time", results$time, sep = "_")
      filename <- paste("../results/", result_name, ".RData", sep = "")
      save(results, file = filename)
      print(paste("Results for L = ", l, " saved in ", filename, sep = ""))
    }
  }
}


#=========================================================================================
# Plotting 
#=========================================================================================

graphics.off()
for (file in rdata_files) { 
  print(file)
  load(file)
  # Extract parameters from filename
  filename <- basename(file)
  filename_parts <- strsplit(filename, "_")[[1]]
  l <- safe_extract("L", filename_parts)
  # Extract file name without extension
  file_base <- tools::file_path_sans_ext(basename(file))
  # Create a folder for saving plots if it doesn't exist
  output_dir <- paste("../print/plot",file_base, sep = "_")  # Change this to your desired folder
  if (!dir.exists(output_dir)) {
    dir.create(output_dir)
  }
  ### First plot - Posterior distribution of the number of clusters
  post_total_cls = table(unlist(results$total_cls))/length(unlist(results$total_cls))
  df <- data.frame(cluster_found = as.numeric(names(post_total_cls)),
                   rel_freq = as.numeric(post_total_cls))

  p1 <- ggplot(data = df, aes(x = factor(cluster_found), y = rel_freq)) + 
    geom_col() + 
    labs(
      x = "Cluster Found",
      y = "Relative Frequency",
      title = paste("Posterior distribution of the number of clusters ( L =", l, ")")
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
      title = paste("Trace of Number of Clusters starting from L =", l)
    ) +
    theme_minimal()
  print(p2)
  ggsave(filename = file.path(output_dir, paste0(substr(file_base,31,60), "_trace_num_clusters.png")), plot = p2, bg = "white")
  
  ### Log-likelihood trace
  log_likelihood_df_bis <- data.frame(
    Iteration = seq_along(results$loglikelihood),
    LogLikelihood = results$loglikelihood
  )
  
  p3_bis <- ggplot(log_likelihood_df_bis, aes(x = Iteration, y = LogLikelihood)) +
    geom_line() +
    labs(
      x = "Iteration",
      y = "Log-Likelihood",
      title = paste("Log-Likelihood of Clusters starting from L =", l)
    ) +
    theme_minimal()
  print(p3_bis)
  ggsave(filename = file.path(output_dir, paste0(substr(file_base,31,60), "_log_likelihood_bfsm.png")), plot = p3_bis, bg = "white")
  
  ### Fourth plot - Posterior similarity matrix
  # Vectorized approach to create the matrix
  C <- matrix(unlist(lapply(results$c_i, function(x) x + 1)), 
              nrow = iterations, 
              ncol = nrow(X), 
              byrow = TRUE)
  
  required_packages <- c("spam", "fields", "viridisLite","RColorBrewer","pheatmap")
  for (pkg in required_packages) {
    if (!require(pkg, character.only = TRUE)) {
      install.packages(pkg)
      library(pkg, character.only = TRUE)
    }
  }
  
  psm = comp.psm(C)
  VI = minVI(psm)
  
  # More informative output
  cat("Cluster Sizes:\n")
  print(table(VI$cl))
  cat("\nAdjusted Rand Index:", arandi(VI$cl, gt), "\n")

  png(filename = file.path(output_dir, paste0(file_base, "matrix.png")), width = 800, height = 800)
  myplotpsm(psm, classes=VI$cl, ax=F, ay=F)
  dev.off()  # Close the device to save the first plot

  # Fifth plot - Auto-correlation plot
  mcmc_list <- list( ncls = unlist(results$total_cls), logl = results$loglikelihood)
  mcmc_matrix <- do.call(cbind, mcmc_list)
  png(filename = file.path(output_dir, paste0(substr(file_base,31,60), "acf.png")), width = 800, height = 800)
  acf(mcmc_matrix)
  dev.off()
}

