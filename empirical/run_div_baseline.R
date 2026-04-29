#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
args_full <- commandArgs(trailingOnly = FALSE)

script_path <- function() {
  file_arg <- args_full[grepl("^--file=", args_full)]
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE))
  }
  stop("Unable to resolve script path from Rscript arguments.")
}

print_usage <- function(status = 1L) {
  cat("Usage: run_div_baseline.R --data-csv PATH --grid-csv PATH --pred-csv PATH [options]\n")
  cat("Options:\n")
  cat("  --num-epochs INT   Number of training epochs (default: 1000)\n")
  cat("  --num-layer INT    Number of layers (default: 4)\n")
  cat("  --lr FLOAT         Learning rate (default: 1e-4)\n")
  cat("  --nsample INT      Prediction Monte Carlo sample size (default: 1000)\n")
  cat("  --seed INT         Random seed (default: 1)\n")
  quit(status = status)
}

parse_args <- function(argv) {
  cfg <- list(
    data_csv = NULL,
    grid_csv = NULL,
    pred_csv = NULL,
    num_epochs = 1000L,
    num_layer = 4L,
    lr = 1e-4,
    nsample = 1000L,
    seed = 1L
  )

  i <- 1
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (key %in% c("-h", "--help")) {
      print_usage(status = 0L)
    } else if (key == "--data-csv") {
      cfg$data_csv <- argv[[i + 1]]
      i <- i + 2
    } else if (key == "--grid-csv") {
      cfg$grid_csv <- argv[[i + 1]]
      i <- i + 2
    } else if (key == "--pred-csv") {
      cfg$pred_csv <- argv[[i + 1]]
      i <- i + 2
    } else if (key == "--num-epochs") {
      cfg$num_epochs <- as.integer(argv[[i + 1]])
      i <- i + 2
    } else if (key == "--num-layer") {
      cfg$num_layer <- as.integer(argv[[i + 1]])
      i <- i + 2
    } else if (key == "--lr") {
      cfg$lr <- as.numeric(argv[[i + 1]])
      i <- i + 2
    } else if (key == "--nsample") {
      cfg$nsample <- as.integer(argv[[i + 1]])
      i <- i + 2
    } else if (key == "--seed") {
      cfg$seed <- as.integer(argv[[i + 1]])
      i <- i + 2
    } else {
      stop("Unknown argument: ", key)
    }
  }

  if (is.null(cfg$data_csv) || is.null(cfg$grid_csv) || is.null(cfg$pred_csv)) {
    print_usage(status = 1L)
  }
  cfg
}

cfg <- parse_args(args)

repo_dir <- normalizePath(file.path(dirname(script_path()), ".."), mustWork = TRUE)
repo_r_libs <- normalizePath(file.path(repo_dir, "R_libs"), mustWork = FALSE)
.libPaths(unique(c(repo_r_libs, .libPaths())))

if (!requireNamespace("DistributionIV", quietly = TRUE)) {
  stop("CRAN package 'DistributionIV' is not available on .libPaths().")
}
if (!requireNamespace("torch", quietly = TRUE)) {
  stop("R package 'torch' is required for DistributionIV.")
}

data_df <- utils::read.csv(cfg$data_csv, stringsAsFactors = FALSE)
grid_df <- utils::read.csv(cfg$grid_csv, stringsAsFactors = FALSE)

required_data_cols <- c("X", "Z", "Y")
missing_data_cols <- setdiff(required_data_cols, colnames(data_df))
if (length(missing_data_cols) > 0) {
  stop("Data CSV missing required columns: ", paste(missing_data_cols, collapse = ", "))
}
if (!"X" %in% colnames(grid_df)) {
  stop("Grid CSV missing required column: X")
}

set.seed(cfg$seed)
div_mod <- DistributionIV::div(
  X = data_df$X,
  Z = data_df$Z,
  Y = data_df$Y,
  epsx_dim = 50,
  epsh_dim = 50,
  epsy_dim = 50,
  num_epochs = cfg$num_epochs,
  num_layer = cfg$num_layer,
  lr = cfg$lr
)

preds <- predict(div_mod, Xtest = grid_df$X, type = "mean", nsample = cfg$nsample)
preds <- as.numeric(preds)

if (length(preds) != nrow(grid_df)) {
  stop("Unexpected DIV prediction length: expected ", nrow(grid_df), ", got ", length(preds))
}
if (any(!is.finite(preds))) {
  stop("DIV predictions contain non-finite values.")
}

pred_df <- data.frame(X = grid_df$X, div_pred = preds)
utils::write.csv(pred_df, cfg$pred_csv, row.names = FALSE)
cat("Saved DIV predictions to:", cfg$pred_csv, "\n")
