symbol = "TLT"

library(glmnet)
#install.packages("forecastML")
library(forecastML)
library(lubridate)
library(quantmod)
library(ggplot2)
library(sys)

set.seed(224)
data <- getSymbols(symbol, src = "yahoo", from = "2015-01-01", to = Sys.time(), calendar = 'UnitedStates/NYSE', auto.assign = FALSE)
tmp = to.daily(data, indexAt = "endof")
data = tmp[,4]
time = index(tmp)

data_stock = data.frame(coredata(data))
data_train <- forecastML::create_lagged_df(data_stock, type = "train", method = "direct",
                                           outcome_col = 1, lookback = 1:100, horizons = 1:12)

windows <- forecastML::create_windows(data_train, window_length = 0)

model_fn <- function(data) {
  x <- as.matrix(data[, -1, drop = FALSE])
  y <- as.matrix(data[, 1, drop = FALSE])
  model_c <- glmnet::cv.glmnet(x, y, nfolds=10, parallel = TRUE)
  penalty <- model_c$lambda.min
  model <- glmnet::glmnet(x,y, alpha=1, lambda = penalty)
}

model_results <- forecastML::train_model(data_train, windows, model_name = "LASSO", model_function = model_fn)

predict_fn <- function(model, data) {
  data_pred <- as.data.frame(predict(model, as.matrix(data)))
}

data_fit <- predict(model_results, prediction_function = list(predict_fn), data = data_train)

residuals <- residuals(data_fit)

data_forecast <- forecastML::create_lagged_df(data_stock, type = "forecast", method = "direct",
                                              outcome_col = 1, lookback = 1:100, horizons = 1:12)

data_forecasts <- predict(model_results, prediction_function = list(predict_fn), data = data_forecast)

data_forecasts <- forecastML::combine_forecasts(data_forecasts)

# set.seed(224)
# data_forecasts <- forecastML::calculate_intervals(data_forecasts, residuals, 
#                                                   levels = seq(.5, .95, .05), times = 200)




plot_data = data.frame(time, index(data_stock), data_stock)
plot_data$type = "Actual"

colnames(plot_data) <- c("Timestamp", "Index","Price", "Type")

#last_friday = Sys.Date() - wday(Sys.Date() + 1)
#next_friday = Sys.Date() + wday(Sys.Date() + 6)

date = seq(Sys.Date() , by = "day", length.out = 12)

to_add = data_forecasts[,4:5]
to_add$time = date
to_add$type = "Predicted"


colnames(to_add) <- c("Index","Price", "Timestamp", "Type")

final_dataset <- rbind(plot_data, to_add)
final_dataset
