#!/usr/bin/env Rscript
# DIV quantile baseline using DistributionIV::div with predict(type = "quantile").
#
# Purpose: Estimate interventional quantiles q_τ(x) using DIV (neural net-based IV).
#
# Method: DIV learns distributions ε_X, ε_Y, ε_H using adversarial training,
#         then samples from the interventional distribution Y|do(X=x).
#
# Output format (consistent with TabPFN runner):
#   - Predictions: s2q_div_{code}_n{n}_seed{seed}_predictions.csv
#   - Summary: s2q_div_{code}_n{n}_seed{seed}_summary.csv (per-tau RMSE)
#
# Dependencies: R packages DistributionIV, torch, jsonlite, readr

suppressPackageStartupMessages({
  library(jsonlite)
  library(readr)
})

script_path <- function() {
  args_full <- commandArgs(trailingOnly = FALSE)
  file_arg <- args_full[grepl("^--file=", args_full)]
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE))
  }
  normalizePath(sys.frames()[[1]]$ofile, mustWork = FALSE)
}

this_script <- script_path()
repo_dir <- normalizePath(file.path(dirname(this_script), "..", ".."), mustWork = TRUE)
default_data_dir <- file.path(repo_dir, "interv_qtl", "IV_datasets")
default_stage1_dir <- file.path(default_data_dir, "stage1_output")
default_stage2_dir <- file.path(default_data_dir, "stage2_output")
default_taus <- c(0.01, 0.025, 0.1, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.9, 0.975, 0.99)

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
    dgp_codes = "A3_B3",
    train_sizes = NULL,
    seeds = NULL,
    taus = default_taus,
    stage1_dir = default_stage1_dir,
    data_dir = default_data_dir,
    output_dir = default_stage2_dir,
    x_grid_mode = "test_quantile",
    x_grid_points = 200,
    x_min = NA_real_,
    x_max = NA_real_,
    mc_samples = 2000L,
    stage2_random_state = 1L,
    nsample = 1000L,
    num_layer = 3L,
    num_epochs = 1000L,
    lr = 1e-3,
    skip_existing = FALSE,
    benchmark_runtime = FALSE,
    benchmark_train = NULL,
    benchmark_test = NULL,
    benchmark_code = NULL,
    benchmark_seed = NULL,
    benchmark_x_grid_mode = "test_quantile",
    benchmark_x_grid_points = 200L,
    benchmark_timer_scope = "fit_predict",
    benchmark_taus = default_taus
  )

  i <- 1
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (key == "--dgp-codes") {
      parsed <- parse_arg_list(argv, i + 1)
      cfg$dgp_codes <- parsed$values
      i <- parsed$next_idx
      next
    } else if (key == "--train-sizes") {
      parsed <- parse_arg_list(argv, i + 1)
      cfg$train_sizes <- as.integer(parsed$values)
      i <- parsed$next_idx
      next
    } else if (key == "--seeds") {
      parsed <- parse_arg_list(argv, i + 1)
      cfg$seeds <- as.integer(parsed$values)
      i <- parsed$next_idx
      next
    } else if (key == "--taus") {
      tau_str <- argv[[i + 1]]
      cfg$taus <- as.numeric(strsplit(tau_str, ",")[[1]])
      i <- i + 2
      next
    } else if (key == "--output-dir") {
      cfg$output_dir <- argv[[i + 1]]
      i <- i + 2
      next
    } else if (key == "--x-grid-mode") {
      cfg$x_grid_mode <- argv[[i + 1]]
      i <- i + 2
      next
    } else if (key == "--x-grid-points") {
      cfg$x_grid_points <- as.integer(argv[[i + 1]])
      i <- i + 2
      next
    } else if (key == "--x-grid-min") {
      cfg$x_min <- as.numeric(argv[[i + 1]])
      i <- i + 2
      next
    } else if (key == "--x-grid-max") {
      cfg$x_max <- as.numeric(argv[[i + 1]])
      i <- i + 2
      next
    } else if (key == "--mc-samples") {
      cfg$mc_samples <- as.integer(argv[[i + 1]])
      i <- i + 2
      next
    } else if (key == "--stage2-random-state") {
      cfg$stage2_random_state <- as.integer(argv[[i + 1]])
      i <- i + 2
      next
    } else if (key == "--nsample") {
      cfg$nsample <- as.integer(argv[[i + 1]])
      i <- i + 2
      next
    } else if (key == "--num-layer") {
      cfg$num_layer <- as.integer(argv[[i + 1]])
      i <- i + 2
      next
    } else if (key == "--num-epochs") {
      cfg$num_epochs <- as.integer(argv[[i + 1]])
      i <- i + 2
      next
    } else if (key == "--lr") {
      cfg$lr <- as.numeric(argv[[i + 1]])
      i <- i + 2
      next
    } else if (key == "--skip-existing") {
      cfg$skip_existing <- TRUE
      i <- i + 1
      next
    } else if (key == "--benchmark-runtime") {
      cfg$benchmark_runtime <- TRUE
      i <- i + 1
      next
    } else if (key == "--benchmark-train") {
      cfg$benchmark_train <- argv[[i + 1]]
      i <- i + 2
      next
    } else if (key == "--benchmark-test") {
      cfg$benchmark_test <- argv[[i + 1]]
      i <- i + 2
      next
    } else if (key == "--benchmark-code") {
      cfg$benchmark_code <- argv[[i + 1]]
      i <- i + 2
      next
    } else if (key == "--benchmark-seed") {
      cfg$benchmark_seed <- as.integer(argv[[i + 1]])
      i <- i + 2
      next
    } else if (key == "--benchmark-x-grid-mode") {
      cfg$benchmark_x_grid_mode <- argv[[i + 1]]
      i <- i + 2
      next
    } else if (key == "--benchmark-x-grid-points") {
      cfg$benchmark_x_grid_points <- as.integer(argv[[i + 1]])
      i <- i + 2
      next
    } else if (key == "--benchmark-timer-scope") {
      cfg$benchmark_timer_scope <- argv[[i + 1]]
      i <- i + 2
      next
    } else if (key == "--benchmark-taus") {
      cfg$benchmark_taus <- as.numeric(strsplit(argv[[i + 1]], ",")[[1]])
      i <- i + 2
      next
    } else {
      stop("Unknown argument: ", key)
    }
  }

  if (isTRUE(cfg$benchmark_runtime)) {
    if (is.null(cfg$benchmark_train) || is.null(cfg$benchmark_test)) {
      stop("Benchmark mode requires --benchmark-train and --benchmark-test.")
    }
    return(cfg)
  }

  if (is.null(cfg$train_sizes) || length(cfg$train_sizes) == 0) {
    stop("--train-sizes is required.")
  }
  if (is.null(cfg$seeds) || length(cfg$seeds) == 0) {
    stop("--seeds is required.")
  }
  cfg
}

validate_timer_scope <- function(scope) {
  scope <- tolower(as.character(scope))
  if (!(scope %in% c("fit_only", "fit_predict", "predict_only"))) {
    stop("Unknown benchmark timer scope: ", scope)
  }
  scope
}

parse_code <- function(code) {
  parts <- strsplit(code, "_")[[1]]
  if (length(parts) != 2) stop("DGP code must be formatted as A?_B?.")
  list(first = toupper(parts[1]), second = toupper(parts[2]))
}

build_x_grid <- function(train_df, test_df, mode, points, x_min, x_max) {
  mode <- tolower(mode)
  if (points <= 1) stop("x_grid_points must be greater than 1.")
  if (mode == "test_quantile") {
    qs <- seq(0.05, 0.95, length.out = points)
    grid <- as.numeric(quantile(test_df$X, probs = qs, names = FALSE))
  } else if (mode == "train_quantile") {
    qs <- seq(0.05, 0.95, length.out = points)
    grid <- as.numeric(quantile(train_df$X, probs = qs, names = FALSE))
  } else if (mode == "train_range") {
    xmin <- if (!is.na(x_min)) x_min else min(train_df$X)
    xmax <- if (!is.na(x_max)) x_max else max(train_df$X)
    if (xmin >= xmax) stop("Invalid x-grid range: min >= max.")
    grid <- seq(from = xmin, to = xmax, length.out = points)
  } else {
    stop("Unknown x_grid_mode: ", mode)
  }
  unique(grid)
}

load_stage1 <- function(path) {
  if (!file.exists(path)) stop("Stage1 CSV not found: ", path)
  df <- readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
  needed <- c("Z", "X", "Y")
  missing <- setdiff(needed, colnames(df))
  if (length(missing) > 0) stop("Stage1 CSV missing columns: ", paste(missing, collapse = ", "))
  df
}

ensure_stage1_csv <- function(code, n, seed, cfg, repo_dir) {
  stage1_csv <- file.path(cfg$stage1_dir, sprintf("iv_stage1_train_%s_n%d_seed%d.csv", code, n, seed))
  if (file.exists(stage1_csv)) {
    return(stage1_csv)
  }

  codes <- parse_code(code)
  train_name <- sprintf("train_data_%s_n%d_seed%d.csv", code, n, seed)
  test_name <- sprintf("test_data_%s.csv", code)
  batch_data_dir <- file.path(repo_dir, "interv_mean", "IV_datasets")
  batch_stage1_csv <- file.path(batch_data_dir, "stage1_output", sprintf("iv_stage1_train_%s_n%d_seed%d.csv", code, n, seed))
  if (file.exists(batch_stage1_csv)) {
    return(batch_stage1_csv)
  }
  if (
    file.exists(file.path(cfg$data_dir, "train", train_name)) &&
    file.exists(file.path(cfg$data_dir, "test", test_name))
  ) {
    base_data_dir <- cfg$data_dir
  } else {
    base_data_dir <- batch_data_dir
  }
  py_bin <- Sys.getenv("RETICULATE_PYTHON", unset = "python")
  script <- tempfile(fileext = ".py")
  stderr_file <- tempfile(fileext = ".err")
  on.exit(unlink(c(script, stderr_file)), add = TRUE)

  py_code <- paste0(
    "import sys\n",
    "sys.path.insert(0, r'", repo_dir, "')\n",
    "sys.path.insert(0, r'", file.path(repo_dir, "tabcf_core"), "')\n",
    "from stage1_control import Stage1Config, run_stage1_experiment\n",
    "cfg = Stage1Config(random_state=1, backend_name='tabpfn')\n",
    sprintf(
      "run_stage1_experiment('%s', '%s', cfg, train_sample_size=%d, seed=%d, output_dir=r'%s', base_dir=r'%s', save_outputs=True, use_timestamp=False)\n",
      codes$first,
      codes$second,
      as.integer(n),
      as.integer(seed),
      normalizePath(cfg$stage1_dir, mustWork = FALSE),
      normalizePath(base_data_dir, mustWork = TRUE)
    )
  )
  writeLines(py_code, con = script, useBytes = TRUE)

  message("Stage1 CSV missing; generating default TabPFN Stage1 for ", code, " n=", n, " seed=", seed)
  res <- tryCatch(
    system2(py_bin, args = script, stdout = TRUE, stderr = stderr_file),
    warning = function(w) w,
    error = function(e) e
  )
  err_output <- tryCatch(readLines(stderr_file, warn = FALSE), error = function(e) character())
  status <- attr(res, "status")
  if (!is.null(status) && status != 0) {
    stop(
      "Stage1 generation failed (status=", status, "):\n",
      paste(err_output, collapse = "\n"),
      "\n",
      paste(res, collapse = "\n")
    )
  }
  if (length(err_output) > 0) {
    writeLines(err_output, con = stderr())
  }
  if (!file.exists(stage1_csv)) {
    stop("Stage1 CSV still missing after generation: ", stage1_csv)
  }
  stage1_csv
}

load_test <- function(path) {
  if (!file.exists(path)) stop("Test CSV not found: ", path)
  df <- readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
  needed <- c("X")
  missing <- setdiff(needed, colnames(df))
  if (length(missing) > 0) stop("Test CSV missing columns: ", paste(missing, collapse = ", "))
  df
}

compute_truth_py <- function(first_stage,
                             second_stage,
                             n,
                             seed,
                             x_grid,
                             taus,
                             mc_samples,
                             random_state,
                             repo_dir) {
  # Call Python's compute_true_quantiles() from R for ground truth consistency.
  #
  # Purpose: Ensure all methods (TabPFN, DIV, IVQR) use identical ground truth.
  #
  # Approach: Generate temporary Python script, execute via system2(), parse JSON output.
  #
  # Returns: Matrix of shape (length(x_grid), length(taus)) with true quantiles.
  
  py_bin <- Sys.getenv("RETICULATE_PYTHON", unset = "python")
  json_x <- jsonlite::toJSON(as.numeric(x_grid), auto_unbox = TRUE)
  json_tau <- jsonlite::toJSON(as.numeric(taus), auto_unbox = TRUE)
  script <- tempfile(fileext = ".py")
  stderr_file <- tempfile(fileext = ".err")
  on.exit(unlink(c(script, stderr_file)), add = TRUE)
  py_code <- paste0(
    "import sys, json, io, contextlib, numpy as np, numpy.random as npr\n",
    "buf = io.StringIO()\n",
    "with contextlib.redirect_stdout(buf):\n",
    "    sys.path.insert(0, r'", repo_dir, "')\n",
    "    sys.path.insert(0, r'", file.path(repo_dir, "interv_qtl"), "')\n",
    "    sys.path.insert(0, r'", file.path(repo_dir, "tabcf_core"), "')\n",
    "    sys.path.insert(0, r'", file.path(repo_dir, "interv_mean"), "')\n",
    "    from dgp import DGPConfig\n",
    "    from cdf_to_quantiles import compute_true_quantiles\n",
    sprintf("    cfg = DGPConfig(n=%d, seed=%d, first_stage='%s', second_stage='%s')\n", n, seed, first_stage, second_stage),
    sprintf("    x_grid = np.array(%s, dtype=float)\n", json_x),
    sprintf("    taus = %s\n", json_tau),
    sprintf("    rng = npr.default_rng(%d)\n", random_state),
    sprintf("    q = compute_true_quantiles(cfg, x_grid, tuple(taus), %d, rng)\n", mc_samples),
    "captured = buf.getvalue()\n",
    "if captured:\n",
    "    sys.stderr.write(captured)\n",
    "json.dump(q.tolist(), sys.stdout)\n"
  )
  writeLines(py_code, con = script, useBytes = TRUE)
  res <- tryCatch(
    system2(py_bin, args = script, stdout = TRUE, stderr = stderr_file),
    warning = function(w) w,
    error = function(e) e
  )
  err_output <- tryCatch(readLines(stderr_file, warn = FALSE), error = function(e) character())
  status <- attr(res, "status")
  if (!is.null(status) && status != 0) {
    stop(
      "Python truth script failed (status=", status, "):\n",
      paste(err_output, collapse = "\n"),
      "\n",
      paste(res, collapse = "\n")
    )
  }
  if (length(err_output) > 0) {
    writeLines(err_output, con = stderr())
  }
  out_text <- paste(res, collapse = "")
  truth <- jsonlite::fromJSON(out_text)
  if (is.null(dim(truth))) {
    truth <- matrix(truth, nrow = length(x_grid), ncol = length(taus))
  }
  truth
}

fit_div_model <- function(train_df, num_layer, num_epochs, lr, random_state) {
  if (!requireNamespace("DistributionIV", quietly = TRUE)) {
    stop("R package 'DistributionIV' is required (install.packages('DistributionIV')).")
  }
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("R package 'torch' is required for DistributionIV.")
  }
  set.seed(as.integer(random_state))
  torch::torch_manual_seed(as.integer(random_state))
  model <- DistributionIV::div(
    X = train_df$X,
    Z = train_df$Z,
    Y = train_df$Y,
    epsx_dim = 50,
    epsy_dim = 50,
    epsh_dim = 50,
    num_epochs = num_epochs,
    num_layer = num_layer,
    lr = lr
  )
}

predict_div_quantile <- function(model, x_grid, taus, nsample) {
  n_x <- length(x_grid)
  n_tau <- length(taus)
  pred_mat <- matrix(NA_real_, nrow = n_x, ncol = n_tau)
  for (j in seq_along(taus)) {
    # DistributionIV::predict expects the quantile level via the 'quantile' argument when type = "quantile".
    res <- predict(
      model,
      Xtest = x_grid,
      type = "quantile",
      quantile = taus[[j]],
      nsample = nsample
    )
    if (is.null(dim(res))) {
      res <- as.numeric(res)
      if (length(res) != n_x) stop("Predict returned length ", length(res), " for tau=", taus[[j]], " expected ", n_x)
    } else if (length(dim(res)) == 2 && nrow(res) == n_x) {
      # DistributionIV may return one draw per column; average to stabilize.
      res <- rowMeans(as.matrix(res))
    } else if (length(dim(res)) == 2 && ncol(res) == n_x) {
      res <- colMeans(as.matrix(res))
    } else {
      res <- as.numeric(res)
      if (length(res) != n_x) stop("Predict returned unexpected shape for tau=", taus[[j]], " (dims=", paste(dim(res), collapse = "x"), ").")
    }
    pred_mat[, j] <- res
  }
  pred_mat
}

fit_div_quantile <- function(train_df, x_grid, taus, nsample, num_layer, num_epochs, lr, random_state) {
  model <- fit_div_model(train_df, num_layer, num_epochs, lr, random_state)
  on.exit({
    model <- NULL
    gc(verbose = FALSE)
  }, add = TRUE)
  predict_div_quantile(model, x_grid, taus, nsample)
}

normalize_pred_matrix <- function(preds, n_x, n_tau) {
  # Accept common shapes and coerce to (n_x, n_tau).
  if (is.list(preds)) {
    preds <- unlist(preds, use.names = FALSE)
  }
  if (is.null(dim(preds))) {
    return(matrix(as.numeric(preds), nrow = n_x, ncol = n_tau))
  }
  if (length(dim(preds)) == 2) {
    if (nrow(preds) == n_x && ncol(preds) == n_tau) {
      return(preds)
    }
    if (nrow(preds) == n_tau && ncol(preds) == n_x) {
      return(t(preds))
    }
  }
  if (length(dim(preds)) == 3) {
    dims <- dim(preds)
    if (dims[1] == n_x && dims[2] == n_tau) {
      return(apply(preds, c(1, 2), mean))
    }
    if (dims[1] == n_tau && dims[2] == n_x) {
      return(t(apply(preds, c(1, 2), mean)))
    }
  }
  flat <- as.numeric(preds)
  if (length(flat) == n_x * n_tau) {
    return(matrix(flat, nrow = n_x, ncol = n_tau))
  }
  stop("Unexpected prediction shape from DistributionIV::predict (dims=", paste(dim(preds), collapse = "x"), ").")
}

emit_benchmark_payload <- function(seconds, status = "ok", error = NULL) {
  payload <- list(
    seconds = as.numeric(seconds),
    status = as.character(status)
  )
  if (!is.null(error) && nzchar(as.character(error))) {
    payload$error <- as.character(error)
  }
  cat(jsonlite::toJSON(payload, auto_unbox = TRUE), "\n")
}

run_benchmark_runtime <- function(cfg) {
  timer_scope <- validate_timer_scope(cfg$benchmark_timer_scope)
  result <- tryCatch({
    train_df <- load_stage1(cfg$benchmark_train)
    test_df <- load_test(cfg$benchmark_test)
    x_grid <- build_x_grid(
      train_df,
      test_df,
      cfg$benchmark_x_grid_mode,
      cfg$benchmark_x_grid_points,
      NA_real_,
      NA_real_
    )

    model <- NULL
    on.exit({
      model <- NULL
      gc(verbose = FALSE)
    }, add = TRUE)

    if (timer_scope == "predict_only") {
      model <- fit_div_model(
        train_df = train_df,
        num_layer = cfg$num_layer,
        num_epochs = cfg$num_epochs,
        lr = cfg$lr,
        random_state = cfg$stage2_random_state
      )
    }

    start_time <- Sys.time()
    if (timer_scope == "fit_only") {
      model <- fit_div_model(
        train_df = train_df,
        num_layer = cfg$num_layer,
        num_epochs = cfg$num_epochs,
        lr = cfg$lr,
        random_state = cfg$stage2_random_state
      )
    } else if (timer_scope == "fit_predict") {
      model <- fit_div_model(
        train_df = train_df,
        num_layer = cfg$num_layer,
        num_epochs = cfg$num_epochs,
        lr = cfg$lr,
        random_state = cfg$stage2_random_state
      )
      preds_raw <- predict_div_quantile(
        model = model,
        x_grid = x_grid,
        taus = cfg$benchmark_taus,
        nsample = cfg$nsample
      )
      preds_mat <- normalize_pred_matrix(preds_raw, length(x_grid), length(cfg$benchmark_taus))
      if (!all(dim(preds_mat) == c(length(x_grid), length(cfg$benchmark_taus)))) {
        stop("DIV benchmark prediction shape mismatch.")
      }
    } else if (timer_scope == "predict_only") {
      preds_raw <- predict_div_quantile(
        model = model,
        x_grid = x_grid,
        taus = cfg$benchmark_taus,
        nsample = cfg$nsample
      )
      preds_mat <- normalize_pred_matrix(preds_raw, length(x_grid), length(cfg$benchmark_taus))
      if (!all(dim(preds_mat) == c(length(x_grid), length(cfg$benchmark_taus)))) {
        stop("DIV benchmark prediction shape mismatch.")
      }
    }
    list(
      seconds = as.numeric(difftime(Sys.time(), start_time, units = "secs")),
      status = "ok",
      error = NULL
    )
  }, error = function(err) {
    list(
      seconds = as.numeric(difftime(Sys.time(), start_time, units = "secs")),
      status = "error",
      error = conditionMessage(err)
    )
  })
  emit_benchmark_payload(result$seconds, result$status, result$error)
}

run_single <- function(code,
                       n,
                       seed,
                       cfg) {
  codes <- parse_code(code)
  first_stage <- codes$first
  second_stage <- codes$second
  stage1_csv <- ensure_stage1_csv(code, n, seed, cfg, repo_dir)
  test_csv <- file.path(cfg$data_dir, "test", sprintf("test_data_%s.csv", code))
  pred_path <- file.path(cfg$output_dir, sprintf("s2q_div_%s_n%d_seed%d_predictions.csv", code, n, seed))
  summary_path <- file.path(cfg$output_dir, sprintf("s2q_div_%s_n%d_seed%d_summary.csv", code, n, seed))

  if (cfg$skip_existing && file.exists(pred_path) && file.exists(summary_path)) {
    message("Skip existing combo: ", code, " n=", n, " seed=", seed)
    return(invisible(NULL))
  }

  train_df <- load_stage1(stage1_csv)
  test_df <- load_test(test_csv)
  x_grid <- build_x_grid(train_df, test_df, cfg$x_grid_mode, cfg$x_grid_points, cfg$x_min, cfg$x_max)

  combo_start <- Sys.time()
  preds_raw <- fit_div_quantile(
    train_df = train_df,
    x_grid = x_grid,
    taus = cfg$taus,
    nsample = cfg$nsample,
    num_layer = cfg$num_layer,
    num_epochs = cfg$num_epochs,
    lr = cfg$lr,
    random_state = cfg$stage2_random_state
  )
  preds_mat <- normalize_pred_matrix(preds_raw, length(x_grid), length(cfg$taus))

  q_true <- compute_truth_py(
    first_stage = first_stage,
    second_stage = second_stage,
    n = n,
    seed = seed,
    x_grid = x_grid,
    taus = cfg$taus,
    mc_samples = cfg$mc_samples,
    random_state = cfg$stage2_random_state,
    repo_dir = repo_dir
  )
  if (!all(dim(q_true) == dim(preds_mat))) {
    stop("Truth and prediction shapes mismatch.")
  }

  sq_err <- (preds_mat - q_true) ^ 2
  mae <- abs(preds_mat - q_true)
  rmse_tau <- sqrt(colMeans(sq_err))
  mae_tau <- colMeans(mae)
  mse_tau <- colMeans(sq_err)

  dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)
  pred_records <- list()
  for (i in seq_along(x_grid)) {
    for (j in seq_along(cfg$taus)) {
      pred_records[[length(pred_records) + 1]] <- list(
        code = code,
        train_size = n,
        seed = seed,
        X = x_grid[[i]],
        tau = cfg$taus[[j]],
        q_pred = preds_mat[i, j],
        q_true = q_true[i, j],
        sq_err = sq_err[i, j]
      )
    }
  }
  pred_df <- do.call(
    rbind,
    lapply(pred_records, function(x) as.data.frame(x, stringsAsFactors = FALSE))
  )
  pred_df$tau <- as.numeric(pred_df$tau)
  pred_df$X <- as.numeric(pred_df$X)
  pred_df$train_size <- as.integer(pred_df$train_size)
  pred_df$seed <- as.integer(pred_df$seed)
  pred_df <- pred_df[order(pred_df$tau, pred_df$X), ]
  readr::write_csv(pred_df, pred_path)

  summary_rows <- list()
  for (j in seq_along(cfg$taus)) {
    summary_rows[[length(summary_rows) + 1]] <- list(
      code = code,
      train_size = n,
      seed = seed,
      tau = cfg$taus[[j]],
      rmse = rmse_tau[[j]],
      mae = mae_tau[[j]],
      mse = mse_tau[[j]],
      n_x_grid = length(x_grid),
      mc_samples = cfg$mc_samples,
      x_grid_mode = cfg$x_grid_mode,
      x_grid_points = cfg$x_grid_points,
      stage2_random_state = cfg$stage2_random_state,
      nsample = cfg$nsample,
      num_layer = cfg$num_layer,
      num_epochs = cfg$num_epochs,
      lr = cfg$lr,
      elapsed_seconds = as.numeric(difftime(Sys.time(), combo_start, units = "secs")),
      stage1_csv = stage1_csv
    )
  }
  summary_df <- do.call(
    rbind,
    lapply(summary_rows, function(x) as.data.frame(x, stringsAsFactors = FALSE))
  )
  readr::write_csv(summary_df, summary_path)
  message("✅ DIV quantile done: ", code, " n=", n, " seed=", seed, " -> ", basename(pred_path))
}

main <- function() {
  cfg <- parse_args()
  if (isTRUE(cfg$benchmark_runtime)) {
    run_benchmark_runtime(cfg)
    return(invisible(NULL))
  }
  message("DIV quantile baseline starting...")
  message("Codes: ", paste(cfg$dgp_codes, collapse = ", "))
  message("Train sizes: ", paste(cfg$train_sizes, collapse = ", "))
  message("Seeds: ", paste(cfg$seeds, collapse = ", "))
  for (code in cfg$dgp_codes) {
    for (n in cfg$train_sizes) {
      for (seed in cfg$seeds) {
        run_single(code, n, seed, cfg)
      }
    }
  }
  message("DIV quantile baseline completed.")
}

main()
