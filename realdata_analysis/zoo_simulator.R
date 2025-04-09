#=========================================================================================
# Library loading 
#=========================================================================================
setwd("~/Documents/Split_and_merge_Gibbs_sampling/realdata_analysis")
library(AntMAN)
library(mcclust.ext)
library(ggplot2)
library(tidyverse)
library(pheatmap)
library(LaplacesDemon)
#library(vcd)
rm(list = ls())
source("../code/old_code/complement_functions.R")

#=========================================================================================
# Load the data and data processing 
#=========================================================================================
zoo <- read.table("../data/zoo.data", h = FALSE, sep = ",")
nam <- zoo$V1
gt <- zoo$V18

zoo <- as.matrix(zoo[, -c(1, 18)] + 1)
zoo[, 13] <- ifelse(zoo[, 13] == 3, 2,
                    ifelse(zoo[, 13] == 5, 3,
                           ifelse(zoo[, 13] == 6, 4,
                                  ifelse(zoo[, 13] == 7, 5,
                                         ifelse(zoo[, 13] == 9, 6,
                                                1)))))
#=========================================================================================
# Chain parameters 
#=========================================================================================

mm <- apply(zoo, 2, function(x) {
  length(table(x))
})
v <- c(rep(6, 12), 3, rep(6,3))
w <- c(rep(0.25, 12), 0.5, rep(0.25, 3))
gamma <- 0.68

## --------------------------------- Info for the chain + thinning ------------------------------------
verbose <- 0
thinning <- 1
sink <- FALSE
sink_file <- "output.txt"

## --------------------------------- Test to conduct info ------------------------------------
# Number of cluster to use as initial assignment 
# Tips: (0 = ground truth, 1 = all together, 101 = all different, else = random assignements with L clusters)
L_plurale <- c(20) 
iterations <- 10000
burnin <- 15000
# Number of latent cluster used into algorithm 8 of Neal (2000)
m <- 3

# Number of refinements for the split and merge algorithm
# t: number of iterations of the restricted Gibbs sampler for the split
# r: number of iterations of the restricted Gibbs sampler for the merge
# Can be vector so the code can be tested for different values of t and r
t_s <- c(10)
r_s <- c(10)
combinations <- expand.grid(t = t_s, r = r_s) # Generate combinations of t and r

# Number of iterations to jump before perfoming a Neal8 step or a split-merge step
# Example: if you want to perform a Neal8 step every 10 iterations and SM every 5 iterations, set steps <- list(c(10, 5)) 
# support different configurations of steps to be performed
steps <- list(c(1, 1)) 

# Convert to desired format (list of vectors)
sam_params <- split(combinations, seq(nrow(combinations)))
sam_params <- lapply(sam_params, function(x) c(x$t, x$r))

# Algorithm to use
n8 <- TRUE
sam <- FALSE

# File name baseline to use w.r.t to the algorithm used
result_name_base <- "Zoo_Test"
if (n8) {
  result_name_base <- paste(result_name_base, "Neal8", sep = "_")
}
if (sam) {
  result_name_base <- paste(result_name_base, "SplitMerge", sep = "_")
}

# Compile the C++ code
Rcpp::sourceCpp("../code/launcher.cpp")

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
      
      # filename construction for the output
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

      # Initial assignments initialization
      if (l == 1) {
        initial_assignments <- rep(0, nrow(zoo))
      } else if (l == 0) {
        initial_assignments <- unlist(gt)
      } else if (l == 101) {
        initial_assignments <- seq(1, nrow(zoo))
      } else {
        initial_assignments <- NULL
      }

      # Running the chain
      results <- run_markov_chain(data = zoo,
                                  attrisize = mm,
                                  gamma = gamma,
                                  v = v,
                                  w = w,
                                  verbose = verbose,
                                  m = m,
                                TRUE  iterations = iterations,
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
      
      # Stop the sink if it was started
      if (sink) {
        sink(NULL)
      }

      # Save the results
      result_name <- paste(result_name, "time", results$time, sep = "_")
      filename <- paste("../results/", result_name, ".RData", sep = "")
      save(results, file = filename)
      print(paste("Results for L = ", l, " saved in ", filename, sep = ""))
    }
  }
}

#=========================================================================================
# Analysis 
#=========================================================================================
# Function to safely extract parameters from filenames 
safe_extract <- function(pattern, filename_parts) {
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

# Function to extract MCMC parameters from results and compute some initial metrics
extract_mcmc_parameters <- function(rdata_files, gt = NULL) {
  results_list <- lapply(rdata_files, function(file) {
    # Load results
    load(file)
    # Extract parameters from filename
    filename <- basename(file)
    filename_parts <- strsplit(filename, "_")[[1]]

    # Compute metrics
    mcmc_list <- list(ncls = unlist(results$total_cls), logl = results$loglikelihood) # nolint: line_length_linter.
    mcmc_matrix <- do.call(cbind, mcmc_list)    
    # Calculate ESS and IAT
    ess <- ESS(mcmc_matrix)
    iat <- IAT(mcmc_matrix)
    # Compute ARI if groundTruth is provided
    ari <- NA_real_
    if (!is.null(gt)) {
      C <- matrix(unlist(lapply(results$c_i, function(x) x + 1)), 
                  nrow = length(results$total_cls), 
                  ncol = length(gt), 
                  byrow = TRUE)
      VI <- minVI(comp.psm(C))
      ari <- arandi(VI$cl, gt)
    }
    # Create data frame
    data.frame(
      filename = filename,
      L = safe_extract("L", filename_parts),
      M = safe_extract("M", filename_parts),
      r = safe_extract("r", filename_parts),
      t = safe_extract("t", filename_parts),
      burnin = safe_extract("BI", filename_parts),
      iterations = safe_extract("IT", filename_parts),
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

# Load the results files from the results directory
results_dir <- file.path(getwd(), "../results")
dir.exists(results_dir)
print(normalizePath(results_dir))
rdata_files <- list.files(results_dir, full.names = TRUE)

mcmc_params <- extract_mcmc_parameters(rdata_files, gt)
print(mcmc_params)

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
      #title = paste("Posterior distribution of the number of clusters ( L =", l, ")")
    ) +
    theme(axis.text.x = element_text(size = 15), axis.text.y = element_text(size = 15), text = element_text(size = 15), 
          panel.background = element_blank(), panel.grid.major = element_line(color = "grey95"),
          panel.grid.minor = element_line(color = "grey95")) +
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
      #title = paste("Trace of Number of Clusters starting from L =", l)
    ) +
    theme(axis.text.x = element_text(size = 15), axis.text.y = element_text(size = 15), text = element_text(size = 15), 
          panel.background = element_blank(), panel.grid.major = element_line(color = "grey95"),
          panel.grid.minor = element_line(color = "grey95"))
  print(p2)
  ggsave(filename = file.path(output_dir, paste0(file_base, "_trace_num_clusters.png")), plot = p2, bg = "white")
  
  ### Third plot - Log-likelihood trace
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
  ggsave(filename = file.path(output_dir, paste0(file_base, "_log_likelihood_bfsm.png")), plot = p3_bis, bg = "white")
  
  ### Fourth plot - Posterior similarity matrix
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
  VI = minVI(psm)
  
  cat("Cluster Sizes:\n")
  print(table(VI$cl))
  cat("\nAdjusted Rand Index:", arandi(VI$cl, gt), "\n")
  
  png(filename = file.path(output_dir, paste0(file_base, "matrix.png")), width = 800, height = 800)
  myplotpsm(psm, classes=VI$cl, ax=F, ay=F)
  dev.off()  # Close the device to save the first plot

  ### Fifth plot - Auto-correlation plot
  mcmc_list <- list( ncls = unlist(results$total_cls), logl = results$loglikelihood)
  mcmc_matrix <- do.call(cbind, mcmc_list)
  png(filename = file.path(output_dir, paste0(file_base, "acf.png")), width = 800, height = 800)
  acf(mcmc_matrix)
  dev.off()
}

