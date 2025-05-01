# Load libraries
library(readr)
library(dplyr)
library(nnet)
library(caret)
library(ggplot2)
library(Metrics)

process <- function(path) {
  # Load the data
  data <- read_csv(path)
  
  # Drop unnecessary columns
  data <- data %>% select(-`PCOS (Y/N)`, -`Sl. No`, -`Patient File No.`)
  y <- read_csv(path)$`PCOS (Y/N)`  # Target variable
  X <- data
  
  # Split into training and test sets
  set.seed(123)
  train_index <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[train_index, ]
  X_test <- X[-train_index, ]
  y_train <- y[train_index]
  y_test <- y[-train_index]
  
  # Train MLP model using nnet
  model <- nnet(as.matrix(X_train), y_train, size = 5, linout = FALSE, skip = FALSE, MaxNWts = 1000, maxit = 500)

  # Predict
  y_pred <- predict(model, as.matrix(X_test), type = "class")
  
  # Accuracy
  accuracy <- mean(y_pred == y_test)
  print(accuracy)
  
  # Save prediction results
  result <- data.frame(ID = 1:length(y_pred), Predicted_Value = y_pred)
  write.csv(result, "results/resultMLP.csv", row.names = FALSE)
  
  # Calculate Metrics
  mse <- mse(y_test, y_pred)
  mae <- mae(y_test, y_pred)
  r2 <- R2(y_pred, y_test)
  rmse_val <- rmse(y_test, y_pred)
  acc <- accuracy
  
  cat("---------------------------------------------------------\n")
  cat(sprintf("MSE VALUE FOR MLP IS %f \n", mse))
  cat(sprintf("MAE VALUE FOR MLP IS %f \n", mae))
  cat(sprintf("R-SQUARED VALUE FOR MLP IS %f \n", r2))
  cat(sprintf("RMSE VALUE FOR MLP IS %f \n", rmse_val))
  cat(sprintf("ACCURACY VALUE MLP IS %f \n", acc))
  cat("---------------------------------------------------------\n")
  
  # Save metrics
  metrics <- data.frame(Parameter = c("MSE", "MAE", "R-SQUARED", "RMSE", "ACCURACY"),
                        Value = c(mse, mae, r2, rmse_val, acc))
  write.csv(metrics, "results/MLPMetrics.csv", row.names = FALSE)
  
  # Plot
  ggplot(metrics, aes(x = Parameter, y = Value, fill = Parameter)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    labs(title = "MLP Metrics Value", x = "Parameter", y = "Value") +
    scale_fill_brewer(palette = "Set2") +
    ggsave("results/MLPMetricsValue.png", width = 8, height = 6)
}

# Example usage:
process("dataset.csv")
