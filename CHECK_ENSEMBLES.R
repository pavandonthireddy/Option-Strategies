library(glmnet)
library(forecastML)
library(lubridate)
library(quantmod)
library(ggplot2)
library(sys)

#' Calculate bootstrap prediction intervals for forecasts
#'
#' The residuals from model training/fit are sampled i.i.d. for (a) each direct forecast
#' horizon for a single time series and (b) each combination of direct forecast horizon and
#' group for multiple time series.
#'
#' @param forecasts A data.frame of forecasts.
#' @param residuals A data.frame of residuals (e.g., \code{residuals(data_fit)})
#' @param index Optional for forecasts from \code{combine_forecasts()}. A string giving the name of the
#' date column in \code{forecasts}.
#' @param outcome Optional for forecasts from \code{combine_forecasts()}. A string giving the name of the
#' forecast column in \code{forecasts}.
#' @param keys Optional. For grouped time series, a character vector giving the column name(s) of the
#' group columns. The key identifies unique time series of residuals for bootstrap sampling. For direct forecasting,
#' a single time series will have one group per direct forecast horizon.
#' @param levels A numeric vector with 1 or more forecast prediction intervals. A level of .95, for example, will
#' return the 0.25 and .975 quantiles of the bootstrapped forecast distribution at each forecast horizon.
#' @param times Integer. The number of bootstrap samples.
#' @param weights Not implemented.
#' @param keep_samples Boolean. If \code{TRUE}, a data.frame of \code{times} bootstrapped forecasts is returned
#' in addition to the calculated forecast prediction intervals. The samples are in the list slot named 'samples'.
#' @return If \code{forecasts} is an object of class 'forecast_results', a \code{forecast_results} object
#' with a new column for each lower- and upper-bound forecast in \code{levels}. If \code{forecasts} is a
#' data.frame, the function return will be the same but without \code{forecastML} attributes. If,
#' \code{keep_samples} is \code{TRUE}, a named list of length 2 is returned with 'forecasts' and 'samples'.
#'
#' @example /R/examples/example_calculate_intervals.R
#' @export
calculate_intervals <- function(forecasts, residuals, index = NULL, outcome = NULL, keys = NULL,
                                levels = c(.95), times = 100L, weights = NULL, keep_samples = FALSE) {
  
  data_residuals <- residuals
  
  #----------------------------------------------------------------------------
  if (methods::is(forecasts, "forecast_results")) {
    
    outcome_levels <- attributes(forecasts)$outcome_levels
    
    if (!is.null(outcome_levels)) {
      stop("Bootstrap prediction intervals are not currently available for factor outcomes.")
    }
    
    names(forecasts)[names(forecasts) == "forecast_period"] <- "index"
    
    index <- "index"
    outcome <- paste0(attributes(forecasts)$outcome_name, "_pred")
  }
  
  if (methods::is(data_residuals, "training_residuals")) {
    
    groups <- attributes(data_residuals)$groups
    
    if (any(unique(forecasts$model) %in% unique(data_residuals$model))) {
      
      keys <- c("model", groups, "model_forecast_horizon")
      
    } else {
      
      keys <- c(groups, "model_forecast_horizon")
      
      data_residuals$model <- NULL
    }
  }
  #----------------------------------------------------------------------------
  if (is.vector(data_residuals)) {
    
    data_residuals <- data.frame("residuals" = data_residuals)
    
  } else {
    
    residuals_name <- names(data_residuals)[!names(data_residuals) %in% c(keys, index, outcome)]
    
    if (length(residuals_name) > 1) {
      stop(paste0("There are extra columns in 'residuals' beyond 'index', 'outcome', 'keys', and the column of residuals.
                  Enter (a) a vector of residuals or, if there are multiple time series, (b) a data.frame with only keys and residuals."))
    }
    
    names(data_residuals)[which(names(data_residuals) == residuals_name)] <- "residuals"
    }
  #----------------------------------------------------------------------------
  if (!is.null(keys)) {
    
    forecasts_merge <- forecasts %>%
      dplyr::filter(!is.na(!!outcome)) %>%
      dplyr::group_by(!!!rlang::syms(keys)) %>%
      dplyr::mutate(".n_samples" = dplyr::n(),
                    ".sample" = 1:dplyr::n())
    
    data_residuals <- dplyr::left_join(data_residuals, forecasts_merge[, c(keys, ".n_samples")], by = keys)
    
    data_residuals <- data_residuals %>%
      dplyr::filter(!is.na(.data$.n_samples)) %>%
      dplyr::filter(!is.na(residuals))
  }
  #----------------------------------------------------------------------------
  forecast_sim <- lapply(seq_len(times), function(i) {
    
    if (!is.null(keys)) {
      
      data_residuals <- data_residuals %>%
        dplyr::group_by(!!!rlang::syms(keys)) %>%
        dplyr::group_modify(.f = function(., ...) {
          data.frame("residuals" = base::sample(.$residuals, .$.n_samples, replace = TRUE))
        }, keep = TRUE) %>%
        dplyr::mutate(".sample" = 1:dplyr::n())
      
      forecasts <- dplyr::left_join(forecasts_merge, data_residuals, by = c(keys, ".sample"))
      
      forecasts$.n_samples <- NULL
      forecasts$.sample <- NULL
      
    } else {
      
      forecasts$residuals <- base::sample(data_residuals$residuals[!is.na(data_residuals$residuals)], nrow(forecasts), replace = TRUE)
    }
    
    forecasts[, outcome] <- forecasts[, outcome] + forecasts[, "residuals"]
    
    forecasts[, "residuals"] <- NULL
    
    forecasts
  })
  
  forecast_sim <- dplyr::bind_rows(forecast_sim, .id = ".times")
  
  forecast_sim <- forecast_sim[, c(keys, index, outcome)]
  #----------------------------------------------------------------------------
  quantiles <- sort(c(.5 + levels / 2, .5 - levels / 2))
  
  quantile_functions <- lapply(seq_along(quantiles), function(i) {
    
    fn <- function(.) stats::quantile(., probs = i)  # 'i' is a placeholder.
    
    body(fn)[[3]] <- quantiles[i]
    
    fn
  })
  
  if (methods::is(forecasts, "forecast_results")) {
    
    names(quantile_functions) <- paste0(outcome, paste0(rep(c("_lower_", "_upper_"), each = length(levels)),
                                                        c(rev(levels), levels) * 100))
    
  } else {
    
    names(quantile_functions) <- paste0(outcome, paste0(rep(c("_pred_lower_", "_pred_upper_"), each = length(levels)),
                                                        c(rev(levels), levels) * 100))
  }
  #----------------------------------------------------------------------------
  forecast_intervals <- forecast_sim %>%
    dplyr::group_by(!!!rlang::syms(keys), index) %>%
    dplyr::summarize_at(outcome, quantile_functions)
  
  data_out <- dplyr::left_join(forecasts, forecast_intervals, by = c(keys, index))
  #----------------------------------------------------------------------------
  if (methods::is(forecasts, "forecast_results")) {
    
    for (attrib in names(attributes(forecasts))[!names(attributes(forecasts)) %in% c("row.names", "names", "class")]) {
      
      attr(data_out, attrib) <- attributes(forecasts)[[attrib]]
      attr(data_out, "prediction_intervals") <- sort(levels)
      class(data_out) <- class(forecasts)
    }
  }
  
  if (keep_samples) {
    
    data_out <- list(data_out, as.data.frame(forecast_sim))
    names(data_out) <- c("forecasts", "samples")
  }
  
  return(data_out)
}

symbol="NVDA"

data <- getSymbols(symbol, src = "yahoo", from = "2015-01-01", to = Sys.time(), calendar = 'UnitedStates/NYSE', auto.assign = FALSE)
tmp = to.weekly(data, indexAt = "endof")
data = tmp[,4]
time = index(tmp)

data_stock = data.frame(coredata(data))
data_train <- forecastML::create_lagged_df(data_stock, type = "train", method = "direct",
                                           outcome_col = 1, lookback = 1:60, horizons = 1:6)

windows <- forecastML::create_windows(data_train, window_length = 0)

model_fn <- function(data) {
  x <- as.matrix(data[, -1, drop = FALSE])
  y <- as.matrix(data[, 1, drop = FALSE])
  model <- glmnet::cv.glmnet(x, y)
}

model_results <- forecastML::train_model(data_train, windows, model_name = "LASSO", model_function = model_fn)

predict_fn <- function(model, data) {
  data_pred <- as.data.frame(predict(model, as.matrix(data)))
}

data_fit <- predict(model_results, prediction_function = list(predict_fn), data = data_train)

residuals <- residuals(data_fit)

data_forecast <- forecastML::create_lagged_df(data_stock, type = "forecast", method = "direct",
                                              outcome_col = 1, lookback = 1:60, horizons = 1:12)

data_forecasts <- predict(model_results, prediction_function = list(predict_fn), data = data_forecast)

data_forecasts <- forecastML::combine_forecasts(data_forecasts)

set.seed(224)
data_forecasts <- calculate_intervals(data_forecasts, residuals, 
                                                  levels = seq(.5, .95, .05), times = 200)




plot_data = data.frame(time, index(data_stock), data_stock)
plot_data$type = "Actual"

colnames(plot_data) <- c("Timestamp", "Index","Price", "Type")

#last_friday = Sys.Date() - wday(Sys.Date() + 1)
next_friday = Sys.Date() + wday(Sys.Date() + 6)

date = seq(next_friday , by = "week", length.out = 12)

to_add = data_forecasts[,4:5]
to_add$time = date
to_add$type = "Predicted"


colnames(to_add) <- c("Index","Price", "Timestamp", "Type")

final_dataset <- rbind(plot_data, to_add)
