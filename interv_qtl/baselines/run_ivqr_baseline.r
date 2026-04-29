#!/usr/bin/env Rscript
# Chernozhukov-Hansen IV quantile regression baseline using the IVQR package.
#
# Purpose: Estimate interventional quantiles q_τ(x) using IVQR (Chernozhukov-Hansen method).
#
# Method: Linear IV quantile regression, estimates coefficients for:
#         q_τ(x) = β₀(τ) + β₁(τ) · x
#
# Specification: Y ~ X | Z | 1  (endogenous X, instrument Z, intercept-only exogenous)
#
# Output format (consistent with TabPFN runner):
#   - Predictions: s2q_ivqr_{code}_n{n}_seed{seed}_predictions.csv
#   - Summary: s2q_ivqr_{code}_n{n}_seed{seed}_summary.csv (per-tau RMSE)
#
# Dependencies: R packages IVQR, Formula, quantreg, jsonlite, readr

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
repo_r_libs <- file.path(repo_dir, "R_libs")
if (dir.exists(repo_r_libs)) {
  .libPaths(unique(c(repo_r_libs, .libPaths())))
}

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
    ivqr_grid_min = NA_real_,
    ivqr_grid_max = NA_real_,
    ivqr_grid_points = 201L,
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
    } else if (key == "--ivqr-grid-min") {
      cfg$ivqr_grid_min <- as.numeric(argv[[i + 1]])
      i <- i + 2
      next
    } else if (key == "--ivqr-grid-max") {
      cfg$ivqr_grid_max <- as.numeric(argv[[i + 1]])
      i <- i + 2
      next
    } else if (key == "--ivqr-grid-points") {
      cfg$ivqr_grid_points <- as.integer(argv[[i + 1]])
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

ensure_ivqr_dependencies <- function() {
  needed <- c("Formula", "quantreg")
  missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop("Missing required R packages for IVQR: ", paste(missing, collapse = ", "),
         ". Install them in R_LIBS or system library.")
  }
}

.ivqr_fun_cached <- NULL
get_ivqr_fun <- function(repo_dir) {
  if (!is.null(.ivqr_fun_cached)) return(.ivqr_fun_cached)

  ivqr_path <- file.path(repo_dir, "IVQR")
  patch_ivqr_vc <- function(fun) {
    safe_vc <- function(object, covariance, bd_rule = "Silver", h_multi = 1) {
      taus <- object$taus
      kd <- ncol(object$PSI)
      len <- length(taus)
      se <- matrix(NA_real_, nrow = kd, ncol = len)
      cov_mats <- array(NA_real_, dim = c(kd, kd, len))
      J_array <- array(NA_real_, dim = c(kd, kd, len))
      vc <- list(se = se, cov_mats = cov_mats, J = J_array)
      class(vc) <- "ivqr.vc"
      vc
    }
    env <- environment(fun)
    try(assign("ivqr.vc", safe_vc, envir = env), silent = TRUE)
    if (!identical(env, .GlobalEnv)) {
      try(assignInNamespace("ivqr.vc", safe_vc, ns = environmentName(env)), silent = TRUE)
    }
    if (!exists("ivqr.vc", envir = env, inherits = FALSE)) {
      assign("ivqr.vc", safe_vc, envir = .GlobalEnv)
    }
  }

  if (requireNamespace("IVQR", quietly = TRUE)) {
    .ivqr_fun_cached <<- get("ivqr", asNamespace("IVQR"))
    patch_ivqr_vc(.ivqr_fun_cached)
    return(.ivqr_fun_cached)
  }

  if (requireNamespace("devtools", quietly = TRUE)) {
    message("IVQR package not installed; loading from source via devtools::load_all(", ivqr_path, ")")
    devtools::load_all(ivqr_path, quiet = TRUE)
    if (requireNamespace("IVQR", quietly = TRUE)) {
      .ivqr_fun_cached <<- get("ivqr", asNamespace("IVQR"))
      patch_ivqr_vc(.ivqr_fun_cached)
      return(.ivqr_fun_cached)
    }
  }

  message("IVQR package not installed; sourcing R files directly from ", ivqr_path)
  source(file.path(ivqr_path, "R", "IVQR.R"))
  source(file.path(ivqr_path, "R", "ks.R"))
  source(file.path(ivqr_path, "R", "data.R"))
  if (exists("ivqr", mode = "function", inherits = TRUE)) {
    .ivqr_fun_cached <<- get("ivqr", mode = "function", inherits = TRUE)
    patch_ivqr_vc(.ivqr_fun_cached)
    return(.ivqr_fun_cached)
  }
  stop("Could not load IVQR::ivqr. Please install the IVQR package.")
}

build_ivqr_grid <- function(train_df, grid_min, grid_max, grid_points) {
  # Build the coefficient search grid for IVQR estimation.
  #
  # Background: IVQR optimizes over a grid of potential slopes β₁.
  #
  # Strategy: Heuristically center the grid around the OLS slope estimate,
  #           with width proportional to Y/X range ratio.
  #
  # Args:
  #   train_df: Training data with columns X, Y
  #   grid_min, grid_max: Optional manual bounds (NA = auto-detect)
  #   grid_points: Number of grid points (should be ≥ 100 for stability)
  #
  # Returns: Numeric vector of length grid_points, spanning [gmin, gmax]
  
  if (grid_points <= 1) stop("ivqr_grid_points must be greater than 1.")
  x_vals <- as.numeric(train_df$X)
  y_vals <- as.numeric(train_df$Y)
  span_x <- diff(range(x_vals))
  span_y <- diff(range(y_vals))
  slope_scale <- if (span_x > 0) span_y / max(span_x, .Machine$double.eps) else 1
  slope_scale <- if (is.finite(slope_scale) && slope_scale > 0) slope_scale else 1
  
  # Estimate center from OLS
  lm_coef <- tryCatch({
    stats::coef(stats::lm(y_vals ~ x_vals))[["x_vals"]]
  }, error = function(e) NA_real_)
  center <- if (is.na(lm_coef)) 0 else lm_coef
  width <- max(abs(center), slope_scale, 1)
  
  # Determine bounds
  gmin <- if (!is.na(grid_min)) grid_min else center - 5 * width
  gmax <- if (!is.na(grid_max)) grid_max else center + 5 * width
  if (!is.finite(gmin)) gmin <- -5
  if (!is.finite(gmax)) gmax <- 5
  if (gmin == gmax) {
    gmin <- gmin - 1
    gmax <- gmax + 1
  } else if (gmin > gmax) {
    tmp <- gmin
    gmin <- gmax
    gmax <- tmp
  }
  seq(from = gmin, to = gmax, length.out = grid_points)
}

fit_ivqr_quantile <- function(train_df, taus, grid, random_state, repo_dir) {
  ensure_ivqr_dependencies()
  suppressPackageStartupMessages(library(Formula))
  ivqr_fun <- get_ivqr_fun(repo_dir)
  if (any(!is.finite(grid))) stop("IVQR grid contains non-finite values.")
  set.seed(as.integer(random_state))
  form <- Formula::as.Formula("Y ~ X | Z | 1")
  fit <- ivqr_fun(
    formula = form,
    taus = taus,
    data = train_df,
    grid = grid,
    gridMethod = "Default",
    ivqrMethod = "iqr",
    qrMethod = "br"
  )
  if (is.null(fit$coef$endg_var) || is.null(fit$coef$exog_var)) {
    stop("IVQR fit did not return coefficient matrices.")
  }
  if (!is.null(fit$error_tau_flag) && any(fit$error_tau_flag)) {
    failed <- taus[which(fit$error_tau_flag)]
    stop("IVQR estimation failed for taus: ", paste(failed, collapse = ", "))
  }
  fit
}

predict_ivqr_on_grid <- function(fit, x_grid) {
  coef_endg <- fit$coef$endg_var
  coef_exog <- fit$coef$exog_var
  if (is.null(dim(coef_endg)) || is.null(dim(coef_exog))) {
    stop("IVQR coefficients are not matrices.")
  }
  if (nrow(coef_endg) != 1) stop("Expected one endogenous regressor; got ", nrow(coef_endg))
  if (nrow(coef_exog) < 1) stop("No exogenous coefficients returned.")
  if (nrow(coef_exog) != 1) stop("Exogenous design should be intercept-only; got ", nrow(coef_exog), " rows.")
  n_tau <- ncol(coef_endg)
  if (ncol(coef_exog) != n_tau) {
    stop("Mismatch between endogenous and exogenous coefficient columns.")
  }
  n_x <- length(x_grid)
  preds <- matrix(NA_real_, nrow = n_x, ncol = n_tau)
  for (j in seq_len(n_tau)) {
    alpha <- as.numeric(coef_endg[, j])
    beta <- as.numeric(coef_exog[, j])
    if (length(beta) < 1) stop("Intercept missing for tau index ", j)
    if (any(!is.finite(alpha)) || any(!is.finite(beta))) {
      stop("Non-finite IVQR coefficient for tau index ", j)
    }
    preds[, j] <- beta[[1]] + alpha[[1]] * x_grid
  }
  preds
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
    ivqr_grid <- build_ivqr_grid(train_df, cfg$ivqr_grid_min, cfg$ivqr_grid_max, cfg$ivqr_grid_points)

    fit <- NULL
    if (timer_scope == "predict_only") {
      fit <- fit_ivqr_quantile(
        train_df = train_df,
        taus = cfg$benchmark_taus,
        grid = ivqr_grid,
        random_state = cfg$stage2_random_state,
        repo_dir = repo_dir
      )
    }

    start_time <- Sys.time()
    if (timer_scope == "fit_only") {
      fit <- fit_ivqr_quantile(
        train_df = train_df,
        taus = cfg$benchmark_taus,
        grid = ivqr_grid,
        random_state = cfg$stage2_random_state,
        repo_dir = repo_dir
      )
    } else if (timer_scope == "fit_predict") {
      fit <- fit_ivqr_quantile(
        train_df = train_df,
        taus = cfg$benchmark_taus,
        grid = ivqr_grid,
        random_state = cfg$stage2_random_state,
        repo_dir = repo_dir
      )
      preds_mat <- predict_ivqr_on_grid(fit, x_grid)
      if (!all(dim(preds_mat) == c(length(x_grid), length(cfg$benchmark_taus)))) {
        stop("IVQR benchmark prediction shape mismatch.")
      }
    } else if (timer_scope == "predict_only") {
      preds_mat <- predict_ivqr_on_grid(fit, x_grid)
      if (!all(dim(preds_mat) == c(length(x_grid), length(cfg$benchmark_taus)))) {
        stop("IVQR benchmark prediction shape mismatch.")
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
  pred_path <- file.path(cfg$output_dir, sprintf("s2q_ivqr_%s_n%d_seed%d_predictions.csv", code, n, seed))
  summary_path <- file.path(cfg$output_dir, sprintf("s2q_ivqr_%s_n%d_seed%d_summary.csv", code, n, seed))

  if (cfg$skip_existing && file.exists(pred_path) && file.exists(summary_path)) {
    message("Skip existing combo: ", code, " n=", n, " seed=", seed)
    return(invisible(NULL))
  }

  train_df <- load_stage1(stage1_csv)
  test_df <- load_test(test_csv)
  x_grid <- build_x_grid(train_df, test_df, cfg$x_grid_mode, cfg$x_grid_points, cfg$x_min, cfg$x_max)
  ivqr_grid <- build_ivqr_grid(train_df, cfg$ivqr_grid_min, cfg$ivqr_grid_max, cfg$ivqr_grid_points)

  combo_start <- Sys.time()
  fit <- fit_ivqr_quantile(
    train_df = train_df,
    taus = cfg$taus,
    grid = ivqr_grid,
    random_state = cfg$stage2_random_state,
    repo_dir = repo_dir
  )
  preds_mat <- predict_ivqr_on_grid(fit, x_grid)

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
  ivqr_grid_min_used <- min(ivqr_grid)
  ivqr_grid_max_used <- max(ivqr_grid)
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
      ivqr_grid_min = ivqr_grid_min_used,
      ivqr_grid_max = ivqr_grid_max_used,
      ivqr_grid_points = length(ivqr_grid),
      elapsed_seconds = as.numeric(difftime(Sys.time(), combo_start, units = "secs")),
      stage1_csv = stage1_csv
    )
  }
  summary_df <- do.call(
    rbind,
    lapply(summary_rows, function(x) as.data.frame(x, stringsAsFactors = FALSE))
  )
  readr::write_csv(summary_df, summary_path)
  message("✅ IVQR quantile done: ", code, " n=", n, " seed=", seed, " -> ", basename(pred_path))
}

main <- function() {
  cfg <- parse_args()
  if (isTRUE(cfg$benchmark_runtime)) {
    run_benchmark_runtime(cfg)
    return(invisible(NULL))
  }
  message("IVQR quantile baseline starting...")
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
  message("IVQR quantile baseline completed.")
}

main()
