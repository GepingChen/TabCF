#!/usr/bin/env Rscript
# Real-data quantile baselines for conditional QR, IVQR, and DIV.

script_path <- function() {
  args_full <- commandArgs(trailingOnly = FALSE)
  file_arg <- args_full[grepl("^--file=", args_full)]
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE))
  }
  normalizePath(sys.frames()[[1]]$ofile, mustWork = FALSE)
}

this_script <- script_path()
repo_dir <- normalizePath(file.path(dirname(this_script), ".."), mustWork = TRUE)
repo_r_libs <- file.path(repo_dir, "R_libs")
if (dir.exists(repo_r_libs)) {
  .libPaths(unique(c(repo_r_libs, .libPaths())))
}

parse_args <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  cfg <- list(
    data_csv = NULL,
    grid_csv = NULL,
    curves_csv = NULL,
    coefficients_csv = NULL,
    runtime_csv = NULL,
    taus = c(0.15, 0.25, 0.50, 0.75, 0.85),
    ivqr_grid_min = NA_real_,
    ivqr_grid_max = NA_real_,
    ivqr_grid_points = 201L,
    div_nsample = 1000L,
    div_num_layer = 4L,
    div_num_epochs = 1000L,
    div_lr = 1e-4,
    div_seed = 1L,
    seed = 1L
  )

  i <- 1
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (key == "--data-csv") {
      cfg$data_csv <- argv[[i + 1]]
      i <- i + 2
      next
    }
    if (key == "--grid-csv") {
      cfg$grid_csv <- argv[[i + 1]]
      i <- i + 2
      next
    }
    if (key == "--curves-csv") {
      cfg$curves_csv <- argv[[i + 1]]
      i <- i + 2
      next
    }
    if (key == "--coefficients-csv") {
      cfg$coefficients_csv <- argv[[i + 1]]
      i <- i + 2
      next
    }
    if (key == "--runtime-csv") {
      cfg$runtime_csv <- argv[[i + 1]]
      i <- i + 2
      next
    }
    if (key == "--taus") {
      cfg$taus <- as.numeric(strsplit(argv[[i + 1]], ",")[[1]])
      i <- i + 2
      next
    }
    if (key == "--ivqr-grid-min") {
      cfg$ivqr_grid_min <- as.numeric(argv[[i + 1]])
      i <- i + 2
      next
    }
    if (key == "--ivqr-grid-max") {
      cfg$ivqr_grid_max <- as.numeric(argv[[i + 1]])
      i <- i + 2
      next
    }
    if (key == "--ivqr-grid-points") {
      cfg$ivqr_grid_points <- as.integer(argv[[i + 1]])
      i <- i + 2
      next
    }
    if (key == "--div-nsample") {
      cfg$div_nsample <- as.integer(argv[[i + 1]])
      i <- i + 2
      next
    }
    if (key == "--div-num-layer") {
      cfg$div_num_layer <- as.integer(argv[[i + 1]])
      i <- i + 2
      next
    }
    if (key == "--div-num-epochs") {
      cfg$div_num_epochs <- as.integer(argv[[i + 1]])
      i <- i + 2
      next
    }
    if (key == "--div-lr") {
      cfg$div_lr <- as.numeric(argv[[i + 1]])
      i <- i + 2
      next
    }
    if (key == "--div-seed") {
      cfg$div_seed <- as.integer(argv[[i + 1]])
      i <- i + 2
      next
    }
    if (key == "--seed") {
      cfg$seed <- as.integer(argv[[i + 1]])
      i <- i + 2
      next
    }
    stop("Unknown argument: ", key)
  }

  needed <- c("data_csv", "grid_csv", "curves_csv", "coefficients_csv", "runtime_csv")
  missing <- needed[vapply(needed, function(name) is.null(cfg[[name]]) || !nzchar(cfg[[name]]), logical(1))]
  if (length(missing) > 0) {
    stop("Missing required arguments: ", paste(missing, collapse = ", "))
  }
  if (length(cfg$taus) == 0) {
    stop("At least one tau is required.")
  }
  if (cfg$ivqr_grid_points <= 1) {
    stop("ivqr_grid_points must be greater than 1.")
  }
  cfg
}

ensure_qr_dependencies <- function() {
  needed <- c("quantreg", "Formula")
  missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop("Missing required R packages: ", paste(missing, collapse = ", "))
  }
}

ensure_div_dependencies <- function() {
  needed <- c("DistributionIV", "torch")
  missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop("Missing required DIV R packages: ", paste(missing, collapse = ", "))
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
      j_array <- array(NA_real_, dim = c(kd, kd, len))
      vc <- list(se = se, cov_mats = cov_mats, J = j_array)
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
  if (grid_points <= 1) stop("ivqr_grid_points must be greater than 1.")

  x_vals <- as.numeric(train_df$X)
  y_vals <- as.numeric(train_df$Y)
  span_x <- diff(range(x_vals))
  span_y <- diff(range(y_vals))
  slope_scale <- if (span_x > 0) span_y / max(span_x, .Machine$double.eps) else 1
  slope_scale <- if (is.finite(slope_scale) && slope_scale > 0) slope_scale else 1

  lm_coef <- tryCatch({
    stats::coef(stats::lm(y_vals ~ x_vals))[["x_vals"]]
  }, error = function(e) NA_real_)
  center <- if (is.na(lm_coef)) 0 else lm_coef
  width <- max(abs(center), slope_scale, 1)

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

coerce_prediction_matrix <- function(preds, n_x, n_tau) {
  if (is.null(dim(preds))) {
    if (n_tau == 1) {
      return(matrix(as.numeric(preds), nrow = n_x, ncol = 1))
    }
    stop("Expected a prediction matrix with ", n_tau, " columns.")
  }
  pred_mat <- as.matrix(preds)
  if (!all(dim(pred_mat) == c(n_x, n_tau))) {
    stop(
      "Unexpected prediction shape: expected (", n_x, ", ", n_tau, "), got (",
      paste(dim(pred_mat), collapse = ", "), ")."
    )
  }
  pred_mat
}

fit_qr_curves <- function(train_df, x_grid, taus) {
  qr_start <- Sys.time()
  fit <- quantreg::rq(Y ~ X, tau = taus, data = train_df, method = "br")
  preds <- predict(fit, newdata = data.frame(X = x_grid))
  pred_mat <- coerce_prediction_matrix(preds, length(x_grid), length(taus))
  coef_mat <- as.matrix(stats::coef(fit))
  if (is.null(dim(coef_mat))) {
    coef_mat <- matrix(as.numeric(coef_mat), nrow = 2, ncol = 1)
  }
  if (!all(dim(coef_mat) == c(2, length(taus)))) {
    stop("Unexpected QR coefficient shape.")
  }
  list(
    pred_mat = pred_mat,
    intercept = as.numeric(coef_mat[1, ]),
    slope = as.numeric(coef_mat[2, ]),
    seconds = as.numeric(difftime(Sys.time(), qr_start, units = "secs"))
  )
}

fit_ivqr_quantile <- function(train_df, taus, grid, random_state, repo_dir) {
  ivqr_fun <- get_ivqr_fun(repo_dir)
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
  if (nrow(coef_endg) != 1) stop("Expected one endogenous regressor.")
  if (nrow(coef_exog) != 1) stop("Expected intercept-only exogenous design.")
  n_tau <- ncol(coef_endg)
  preds <- matrix(NA_real_, nrow = length(x_grid), ncol = n_tau)
  for (j in seq_len(n_tau)) {
    slope <- as.numeric(coef_endg[1, j])
    intercept <- as.numeric(coef_exog[1, j])
    if (!is.finite(slope) || !is.finite(intercept)) {
      stop("Non-finite IVQR coefficient for tau index ", j)
    }
    preds[, j] <- intercept + slope * x_grid
  }
  list(
    pred_mat = preds,
    intercept = as.numeric(coef_exog[1, ]),
    slope = as.numeric(coef_endg[1, ])
  )
}

fit_div_model <- function(train_df, num_layer, num_epochs, lr, random_state) {
  ensure_div_dependencies()
  set.seed(as.integer(random_state))
  torch::torch_manual_seed(as.integer(random_state))
  DistributionIV::div(
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

make_curve_rows <- function(method, family, taus, x_grid, pred_mat) {
  rows <- list()
  for (j in seq_along(taus)) {
    for (i in seq_along(x_grid)) {
      rows[[length(rows) + 1]] <- data.frame(
        method = method,
        estimand_family = family,
        tau = as.numeric(taus[[j]]),
        x_log = as.numeric(x_grid[[i]]),
        q_pred_log = as.numeric(pred_mat[i, j]),
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

make_coefficient_rows <- function(method, family, taus, intercept, slope) {
  data.frame(
    method = method,
    estimand_family = family,
    tau = as.numeric(taus),
    intercept_log = as.numeric(intercept),
    slope_log = as.numeric(slope),
    stringsAsFactors = FALSE
  )
}

main <- function() {
  cfg <- parse_args()
  ensure_qr_dependencies()

  train_df <- read.csv(cfg$data_csv, stringsAsFactors = FALSE)
  grid_df <- read.csv(cfg$grid_csv, stringsAsFactors = FALSE)

  needed_data_cols <- c("X", "Y", "Z")
  missing_data <- setdiff(needed_data_cols, names(train_df))
  if (length(missing_data) > 0) {
    stop("Data CSV missing required columns: ", paste(missing_data, collapse = ", "))
  }
  if (!("x_log" %in% names(grid_df))) {
    stop("Grid CSV must contain x_log column.")
  }

  x_grid <- as.numeric(grid_df$x_log)
  taus <- as.numeric(cfg$taus)

  qr_fit <- fit_qr_curves(train_df, x_grid, taus)

  ivqr_grid <- build_ivqr_grid(
    train_df,
    cfg$ivqr_grid_min,
    cfg$ivqr_grid_max,
    cfg$ivqr_grid_points
  )
  ivqr_start <- Sys.time()
  ivqr_fit <- fit_ivqr_quantile(
    train_df = train_df,
    taus = taus,
    grid = ivqr_grid,
    random_state = cfg$seed,
    repo_dir = repo_dir
  )
  ivqr_pred <- predict_ivqr_on_grid(ivqr_fit, x_grid)
  ivqr_seconds <- as.numeric(difftime(Sys.time(), ivqr_start, units = "secs"))

  div_start <- Sys.time()
  div_model <- fit_div_model(
    train_df = train_df,
    num_layer = cfg$div_num_layer,
    num_epochs = cfg$div_num_epochs,
    lr = cfg$div_lr,
    random_state = cfg$div_seed
  )
  div_pred <- predict_div_quantile(
    model = div_model,
    x_grid = x_grid,
    taus = taus,
    nsample = cfg$div_nsample
  )
  div_seconds <- as.numeric(difftime(Sys.time(), div_start, units = "secs"))
  div_model <- NULL
  gc(verbose = FALSE)

  curves_df <- rbind(
    make_curve_rows("QR", "conditional_quantile", taus, x_grid, qr_fit$pred_mat),
    make_curve_rows("IVQR", "interventional_quantile", taus, x_grid, ivqr_pred$pred_mat),
    make_curve_rows("DIV", "interventional_quantile", taus, x_grid, div_pred)
  )
  curves_df <- curves_df[order(curves_df$method, curves_df$tau, curves_df$x_log), ]

  coefficients_df <- rbind(
    make_coefficient_rows("QR", "conditional_quantile", taus, qr_fit$intercept, qr_fit$slope),
    make_coefficient_rows("IVQR", "interventional_quantile", taus, ivqr_pred$intercept, ivqr_pred$slope)
  )
  coefficients_df <- coefficients_df[order(coefficients_df$method, coefficients_df$tau), ]

  runtime_df <- data.frame(
    method = c("QR", "IVQR", "DIV"),
    backend_name = c("", "", ""),
    estimand_family = c("conditional_quantile", "interventional_quantile", "interventional_quantile"),
    seconds = c(qr_fit$seconds, ivqr_seconds, div_seconds),
    stringsAsFactors = FALSE
  )
  runtime_df <- runtime_df[order(runtime_df$seconds, decreasing = TRUE), ]

  write.csv(curves_df, cfg$curves_csv, row.names = FALSE)
  write.csv(coefficients_df, cfg$coefficients_csv, row.names = FALSE)
  write.csv(runtime_df, cfg$runtime_csv, row.names = FALSE)

  message("✅ Real-data quantile QR/IVQR baselines completed.")
}

main()
