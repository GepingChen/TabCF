#!/usr/bin/env Rscript
# Run R baselines (linear IV, nonlinear IV, DIV variants) on bridged sec5.1 datasets.
# Uses the same manifest/results layout as run_py_baselines.py.

args_full <- commandArgs(trailingOnly = FALSE)
script_path <- function() {
  file_arg <- args_full[grepl("^--file=", args_full)]
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE))
  }
  normalizePath(sys.frames()[[1]]$ofile, mustWork = FALSE)
}

this_script <- script_path()
repo_dir <- normalizePath(file.path(dirname(this_script), "..", "..", ".."), mustWork = TRUE)
interv_mean_dir <- file.path(repo_dir, "interv_mean", "interv_mean")
default_manifest <- file.path(interv_mean_dir, "manifests", "generated", "paper_main_s10.json")
default_bridge_dir <- file.path(interv_mean_dir, "io", "bridge")
default_results_dir <- file.path(interv_mean_dir, "io", "results")
default_r_libs <- file.path(repo_dir, "R_libs")

# Make sure the repo-managed R library path is visible before loading packages.
if (dir.exists(default_r_libs)) {
  .libPaths(unique(c(default_r_libs, .libPaths())))
  existing_rlibs <- Sys.getenv("R_LIBS")
  merged <- if (nzchar(existing_rlibs)) paste(default_r_libs, existing_rlibs, sep = ":") else default_r_libs
  Sys.setenv(R_LIBS = merged)
}

suppressPackageStartupMessages({
  library(jsonlite)
  library(readr)
  library(splines)
  library(MASS)
})

print_usage <- function() {
  cat("Usage: run_r_baselines.r [options]\n")
  cat("Options:\n")
  cat("  --manifest PATH        Path to sec5.1 manifest JSON (default: ", default_manifest, ")\n", sep = "")
  cat("  --bridge-dir PATH      Directory containing bridged CSVs (default: ", default_bridge_dir, ")\n", sep = "")
  cat("  --results-dir PATH     Directory to write metrics/predictions (default: ", default_results_dir, ")\n", sep = "")
  cat("  --models m1 [m2 ...]   Subset of models: linear_iv, nonlinear_iv, div, div_2\n")
  cat("  --codes c1 [c2 ...]    Optional subset of codes (e.g., A3_B3)\n")
  cat("  --sizes n1 [n2 ...]    Optional subset of train sizes\n")
  cat("  --seeds s1 [s2 ...]    Optional subset of seeds\n")
  cat("  --overwrite            Recompute even if metrics exist\n")
  cat("  --save-predictions     Save per-sample predictions alongside metrics\n")
  cat("  --benchmark-runtime    Run single-method pure fit/predict timing mode\n")
  cat("  --benchmark-model M    Single benchmark model: linear_iv, nonlinear_iv, div_2\n")
  cat("  --benchmark-train P    Benchmark bridged train CSV path\n")
  cat("  --benchmark-test P     Benchmark bridged test CSV path\n")
  cat("  --benchmark-seed S     Benchmark seed\n")
  cat("  -h, --help             Show this help message\n")
  quit(status = 0)
}

parse_arg_list <- function(argv, start_idx) {
  values <- character()
  i <- start_idx
  while (i <= length(argv) && !startsWith(argv[[i]], "--")) {
    values <- c(values, argv[[i]])
    i <- i + 1
  }
  list(values = values, next_idx = i)
}

parse_args <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  cfg <- list(
    manifest = default_manifest,
    bridge_dir = default_bridge_dir,
    results_dir = default_results_dir,
    models = c("linear_iv", "nonlinear_iv", "div", "div_2"),
    codes = NULL,
    sizes = NULL,
    seeds = NULL,
    overwrite = FALSE,
    save_predictions = FALSE,
    benchmark_runtime = FALSE,
    benchmark_model = NULL,
    benchmark_train = NULL,
    benchmark_test = NULL,
    benchmark_seed = NULL
  )

  i <- 1
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (key %in% c("-h", "--help")) {
      print_usage()
    } else if (key == "--manifest") {
      cfg$manifest <- argv[[i + 1]]
      i <- i + 2
      next
    } else if (key == "--bridge-dir") {
      cfg$bridge_dir <- argv[[i + 1]]
      i <- i + 2
      next
    } else if (key == "--results-dir") {
      cfg$results_dir <- argv[[i + 1]]
      i <- i + 2
      next
    } else if (key == "--models") {
      parsed <- parse_arg_list(argv, i + 1)
      if (length(parsed$values) == 0) stop("--models requires at least one model name.")
      cfg$models <- tolower(parsed$values)
      i <- parsed$next_idx
      next
    } else if (key == "--codes") {
      parsed <- parse_arg_list(argv, i + 1)
      cfg$codes <- parsed$values
      i <- parsed$next_idx
      next
    } else if (key == "--sizes") {
      parsed <- parse_arg_list(argv, i + 1)
      cfg$sizes <- as.integer(parsed$values)
      i <- parsed$next_idx
      next
    } else if (key == "--seeds") {
      parsed <- parse_arg_list(argv, i + 1)
      cfg$seeds <- as.integer(parsed$values)
      i <- parsed$next_idx
      next
    } else if (key == "--overwrite") {
      cfg$overwrite <- TRUE
      i <- i + 1
      next
    } else if (key == "--save-predictions") {
      cfg$save_predictions <- TRUE
      i <- i + 1
      next
    } else if (key == "--benchmark-runtime") {
      cfg$benchmark_runtime <- TRUE
      i <- i + 1
      next
    } else if (key == "--benchmark-model") {
      cfg$benchmark_model <- tolower(argv[[i + 1]])
      i <- i + 2
      next
    } else if (key == "--benchmark-train") {
      cfg$benchmark_train <- argv[[i + 1]]
      i <- i + 2
      next
    } else if (key == "--benchmark-test") {
      cfg$benchmark_test <- argv[[i + 1]]
      i <- i + 2
      next
    } else if (key == "--benchmark-seed") {
      cfg$benchmark_seed <- as.integer(argv[[i + 1]])
      i <- i + 2
      next
    } else {
      stop("Unknown argument: ", key)
    }
  }

  cfg
}

load_manifest <- function(path) {
  if (!file.exists(path)) stop("Manifest not found at: ", path)
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

repo_from_rel <- function(path_rel) {
  normalizePath(file.path(repo_dir, path_rel), mustWork = FALSE)
}

resolve_run_paths <- function(run, bridge_dir) {
  # v2 schema: train_rel/test_rel + bridge_*_name
  if (!is.null(run$train_rel) && !is.null(run$test_rel)) {
    train_path <- repo_from_rel(run$train_rel)
    test_path <- repo_from_rel(run$test_rel)
    if (!is.null(run$bridge_train_name) && !is.null(run$bridge_test_name)) {
      bridge_train <- file.path(bridge_dir, run$bridge_train_name)
      bridge_test <- file.path(bridge_dir, run$bridge_test_name)
    } else {
      stop("Manifest run missing bridge_train_name/bridge_test_name for v2 schema.")
    }
    return(list(train = train_path, test = test_path, bridge_train = bridge_train, bridge_test = bridge_test))
  }

  # Legacy schema fallback
  train_path <- run$train_path
  test_path <- run$test_path
  if (is.null(run$bridge_train_path) || is.null(run$bridge_test_path)) {
    stop("Legacy run missing bridge_train_path/bridge_test_path.")
  }
  bridge_train <- file.path(bridge_dir, basename(run$bridge_train_path))
  bridge_test <- file.path(bridge_dir, basename(run$bridge_test_path))
  list(train = train_path, test = test_path, bridge_train = bridge_train, bridge_test = bridge_test)
}

ensure_cols <- function(df, cols, label) {
  missing <- setdiff(cols, colnames(df))
  if (length(missing) > 0) {
    stop(label, " missing columns: ", paste(missing, collapse = ", "))
  }
}

load_bridged <- function(train_path, test_path) {
  train_df <- readr::read_csv(train_path, show_col_types = FALSE, progress = FALSE)
  test_df <- readr::read_csv(test_path, show_col_types = FALSE, progress = FALSE)
  ensure_cols(train_df, c("X", "Y", "Z"), "Train data")
  ensure_cols(test_df, c("Xint", "mean_int"), "Test data")
  list(train = train_df, test = test_df)
}

cf_lin_nonlin <- function(data_train, data_test, df = 5) {
  step1_lin <- stats::lm(X ~ Z, data = data_train)
  resid_lin <- data_train$X - stats::predict(step1_lin)
  step2_lin <- stats::lm(Y ~ X + resid_lin, data = data_train)

  step1_nonlin <- stats::lm(X ~ splines::ns(Z, df = df), data = data_train)
  resid_nonlin <- data_train$X - stats::predict(step1_nonlin)
  step2_nonlin <- stats::lm(Y ~ splines::ns(X, df = df) + resid_nonlin, data = data_train)

  y_hat_cf_lin <- stats::predict(
    step2_lin,
    newdata = data.frame(X = data_test$Xint, resid_lin = rep(0, length(data_test$Xint)))
  )
  y_hat_cf_nonlin <- stats::predict(
    step2_nonlin,
    newdata = data.frame(X = data_test$Xint, resid_nonlin = rep(0, length(data_test$Xint)))
  )

  list(
    linear_pred = as.numeric(y_hat_cf_lin),
    nonlinear_pred = as.numeric(y_hat_cf_nonlin)
  )
}

run_linear_iv_only <- function(data_train, data_test) {
  step1_lin <- stats::lm(X ~ Z, data = data_train)
  resid_lin <- data_train$X - stats::predict(step1_lin)
  step2_lin <- stats::lm(Y ~ X + resid_lin, data = data_train)
  preds <- stats::predict(
    step2_lin,
    newdata = data.frame(X = data_test$Xint, resid_lin = rep(0, length(data_test$Xint)))
  )
  list(preds = as.numeric(preds), variant = "control_function_linear")
}

run_nonlinear_iv_only <- function(data_train, data_test, df = 5) {
  step1_nonlin <- stats::lm(X ~ splines::ns(Z, df = df), data = data_train)
  resid_nonlin <- data_train$X - stats::predict(step1_nonlin)
  step2_nonlin <- stats::lm(Y ~ splines::ns(X, df = df) + resid_nonlin, data = data_train)
  preds <- stats::predict(
    step2_nonlin,
    newdata = data.frame(X = data_test$Xint, resid_nonlin = rep(0, length(data_test$Xint)))
  )
  list(preds = as.numeric(preds), variant = sprintf("control_function_spline_df%d", df))
}

ensure_distributioniv <- function(pkg_path, require_cran = FALSE) {
  if (!requireNamespace("DistributionIV", quietly = TRUE)) {
    stop("DistributionIV not installed or not on R_LIBS. Install the package (e.g., from: ", pkg_path, ").")
  }
  if (require_cran) {
    desc <- tryCatch(utils::packageDescription("DistributionIV"), error = function(e) NULL)
    repo <- if (!is.null(desc)) desc$Repository else NA_character_
    repo_is_cran <- is.character(repo) && length(repo) == 1 && !is.na(repo) &&
      grepl("CRAN", repo, ignore.case = TRUE)
    if (is.null(desc) || !repo_is_cran) {
      stop(
        "Model div_2 requires DistributionIV installed from CRAN (install.packages('DistributionIV')). ",
        "Current install repository: ",
        if (is.character(repo) && length(repo) == 1 && !is.na(repo)) repo else "<unknown>"
      )
    }
  }
  invisible(TRUE)
}

run_div_impl <- function(train_df,
                         test_df,
                         seed,
                         repo_dir,
                         num_layer = 3,
                         num_epochs = 1000,
                         require_cran = FALSE,
                         variant_prefix = "div") {
  pkg_path <- file.path(repo_dir, "DIV-main", "R", "sec5.3", "DIV_flin")
  ensure_distributioniv(pkg_path, require_cran = require_cran)
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("R package 'torch' is required for DIV.")
  }
  set.seed(seed)
  div_mod <- DistributionIV::div(
    X = train_df$X,
    Z = train_df$Z,
    Y = train_df$Y,
    epsx_dim = 50,
    epsy_dim = 50,
    epsh_dim = 50,
    num_epochs = num_epochs,
    num_layer = num_layer,
    lr = 1e-3
  )
  # predict.DIV is not exported, so use generic predict() after load_all attach.
  preds <- predict(div_mod, Xtest = test_df$Xint, type = "mean", nsample = 1000)
  div_mod <- NULL
  gc(verbose = FALSE)
  list(
    preds = as.numeric(preds),
    variant = sprintf("%s_layer%d_epoch%d", variant_prefix, num_layer, num_epochs)
  )
}

run_div <- function(train_df, test_df, seed, repo_dir, num_layer = 3, num_epochs = 1000) {
  run_div_impl(
    train_df = train_df,
    test_df = test_df,
    seed = seed,
    repo_dir = repo_dir,
    num_layer = num_layer,
    num_epochs = num_epochs,
    require_cran = FALSE,
    variant_prefix = "div"
  )
}

run_div_cran <- function(train_df, test_df, seed, repo_dir, num_layer = 3, num_epochs = 1000) {
  run_div_impl(
    train_df = train_df,
    test_df = test_df,
    seed = seed,
    repo_dir = repo_dir,
    num_layer = num_layer,
    num_epochs = num_epochs,
    require_cran = TRUE,
    variant_prefix = "div_cran"
  )
}

compute_mse <- function(preds, target) {
  mean((as.numeric(preds) - as.numeric(target))^2)
}

run_benchmark_runtime <- function(cfg) {
  if (!isTRUE(cfg$benchmark_runtime)) {
    stop("run_benchmark_runtime() requires benchmark_runtime=TRUE.")
  }
  if (is.null(cfg$benchmark_model) || is.null(cfg$benchmark_train) || is.null(cfg$benchmark_test) || is.null(cfg$benchmark_seed)) {
    stop("Benchmark mode requires --benchmark-model, --benchmark-train, --benchmark-test, and --benchmark-seed.")
  }

  model <- tolower(cfg$benchmark_model)
  if (!(model %in% c("linear_iv", "nonlinear_iv", "div_2"))) {
    stop("Benchmark mode only supports linear_iv, nonlinear_iv, and div_2.")
  }

  data <- load_bridged(cfg$benchmark_train, cfg$benchmark_test)
  elapsed <- proc.time()[["elapsed"]]
  if (model == "linear_iv") {
    result <- run_linear_iv_only(data$train, data$test)
  } else if (model == "nonlinear_iv") {
    result <- run_nonlinear_iv_only(data$train, data$test, df = 5)
  } else {
    result <- run_div_cran(data$train, data$test, seed = cfg$benchmark_seed, repo_dir = repo_dir)
  }
  elapsed <- proc.time()[["elapsed"]] - elapsed

  payload <- list(
    model = model,
    variant = result$variant,
    seconds = as.numeric(elapsed)
  )
  cat(jsonlite::toJSON(payload, auto_unbox = TRUE), "\n", sep = "")
}

main <- function() {
  cfg <- parse_args()
  if (isTRUE(cfg$benchmark_runtime)) {
    run_benchmark_runtime(cfg)
    return(invisible(NULL))
  }
  manifest <- load_manifest(cfg$manifest)
  runs <- manifest$runs
  if (is.null(runs) || length(runs) == 0) stop("Manifest contained no runs.")

  run_filter <- function(run) {
    code_ok <- is.null(cfg$codes) || run$code %in% cfg$codes
    size_ok <- is.null(cfg$sizes) || as.integer(run$train_size) %in% cfg$sizes
    seed_ok <- is.null(cfg$seeds) || as.integer(run$seed) %in% cfg$seeds
    code_ok && size_ok && seed_ok
  }
  runs <- Filter(run_filter, runs)
  if (length(runs) == 0) stop("No runs matched the provided filters.")

  dir.create(cfg$results_dir, recursive = TRUE, showWarnings = FALSE)
  message("Library paths: ", paste(.libPaths(), collapse = " | "))

  total <- length(runs)
  for (idx in seq_along(runs)) {
    run <- runs[[idx]]
    code <- run$code
    size <- as.integer(run$train_size)
    seed <- as.integer(run$seed)
    resolved <- resolve_run_paths(run, cfg$bridge_dir)
    bridge_train <- resolved$bridge_train
    bridge_test <- resolved$bridge_test
    if (!file.exists(bridge_train) || !file.exists(bridge_test)) {
      stop("Missing bridged CSVs for ", code, ", n=", size, ", seed=", seed)
    }

    message(sprintf("[%d/%d] %s n=%d seed=%d", idx, total, code, size, seed))
    data <- load_bridged(bridge_train, bridge_test)
    x_test <- data$test$Xint
    target <- data$test$mean_int
    cf_cache <- NULL
    div_cache <- NULL
    div2_cache <- NULL

    for (model in cfg$models) {
      metrics_path <- file.path(cfg$results_dir, sprintf("%s_%s_n%d_seed%d.json", model, code, size, seed))
      preds_path <- file.path(cfg$results_dir, sprintf("%s_%s_n%d_seed%d_pred.csv", model, code, size, seed))
      if (!cfg$overwrite && file.exists(metrics_path)) {
        message("   Skip ", model, " (metrics exist).")
        next
      }

      model_start <- Sys.time()
      if (model == "linear_iv" || model == "nonlinear_iv") {
        if (is.null(cf_cache)) {
          cf_cache <- cf_lin_nonlin(data$train, data$test, df = 5)
        }
        if (model == "linear_iv") {
          preds <- cf_cache$linear_pred
          variant <- "control_function_linear"
        } else {
          preds <- cf_cache$nonlinear_pred
          variant <- "control_function_spline_df5"
        }
      } else if (model == "div") {
        if (is.null(div_cache)) {
          div_cache <- run_div(data$train, data$test, seed = seed, repo_dir = repo_dir)
        }
        preds <- div_cache$preds
        variant <- div_cache$variant
      } else if (model == "div_2") {
        if (is.null(div2_cache)) {
          div2_cache <- run_div_cran(data$train, data$test, seed = seed, repo_dir = repo_dir)
        }
        preds <- div2_cache$preds
        variant <- div2_cache$variant
      } else {
        stop("Unsupported model: ", model)
      }

      mse_val <- compute_mse(preds, target)
      payload <- list(
        model = model,
        variant = variant,
        code = code,
        scenario = run$scenario,
        train_size = size,
        seed = seed,
        bridge_train = bridge_train,
        bridge_test = bridge_test,
        mse_vs_mean_int = mse_val,
        n_train = nrow(data$train),
        n_test = nrow(data$test)
      )
      jsonlite::write_json(payload, metrics_path, auto_unbox = TRUE, pretty = TRUE)
      if (cfg$save_predictions) {
        pred_df <- data.frame(Xint = x_test, mean_int = target, pred = preds)
        readr::write_csv(pred_df, preds_path)
      }
      elapsed <- as.numeric(difftime(Sys.time(), model_start, units = "secs"))
      message(
        sprintf(
          "   ✅ %s %s n=%d seed=%d done in %.2fs at %s (mse=%.6f)",
          model,
          code,
          size,
          seed,
          elapsed,
          format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
          mse_val
        )
      )
    }
  }
  message("Baseline runs completed.")
}

main()
