process <- function(path) {
  library(class)          # For knn()
  library(readr)          # For reading CSV
  library(caret)          # For data splitting and accuracy
  library(dplyr)          # For data manipulation
  library(Metrics)        # For mse, mae, rmse
  library(ggplot2)        # For plotting

  # Create results folder if it doesn't exist
  if (!dir.exists("results")) dir.create("results")

  # Read the dataset
  data <- read_csv(path)

  # Preprocess: remove unnecessary columns
  data <- data %>% select(-`Sl. No`, -`Patient File No.`)

  # Define features and target
  X <- data %>% select(-`PCOS (Y/N)`)
  y <- data$`PCOS (Y/N)`

  # Normalize features
  normalize <- function(x) { return((x - min(x)) / (max(x) - min(x))) }
  X_norm <- as.data.frame(lapply(X, normalize))

  # Train-test split (89.9% train, 10.1% test)
  set.seed(42)
  trainIndex <- createDataPartition(y, p = 0.899, list = FALSE)
  X_train <- X_norm[trainIndex, ]
  X_test <- X_norm[-trainIndex, ]
  y_train <- y[trainIndex]
  y_test <- y[-trainIndex]

  # Train KNN model (k = 7)
  y_pred <- knn(train = X_train, test = X_test, cl = y_train, k = 7)

  # Convert predictions to numeric
  y_pred_num <- as.numeric(as.character(y_pred))

  # Evaluation metrics
  mse_val <- mse(y_test, y_pred_num)
  mae_val <- mae(y_test, y_pred_num)
  r2_val <- R2_Score(y_pred_num, y_test)
  rmse_val <- rmse(y_test, y_pred_num)
  acc_val <- sum(y_pred == y_test) / length(y_test)

  cat(sprintf("MSE VALUE FOR KNN IS %f\n", mse_val))
  cat(sprintf("MAE VALUE FOR KNN IS %f\n", mae_val))
  cat(sprintf("R-SQUARED VALUE FOR KNN IS %f\n", r2_val))
  cat(sprintf("RMSE VALUE FOR KNN IS %f\n", rmse_val))
  cat(sprintf("ACCURACY VALUE KNN IS %f\n", acc_val))
  cat("---------------------------------------------------------\n")

  # Write predictions
  write_csv(data.frame(ID = 1:length(y_pred), Predicted_Value = y_pred), "results/resultKNN.csv")

  # Write metrics
  metrics_df <- data.frame(
    Parameter = c("MSE", "MAE", "R-SQUARED", "RMSE", "ACCURACY"),
    Value = c(mse_val, mae_val, r2_val, rmse_val, acc_val)
  )
  write_csv(metrics_df, "results/KNNMetrics.csv")

  # Bar plot of metrics
  ggplot(metrics_df, aes(x = Parameter, y = Value, fill = Parameter)) +
    geom_bar(stat = "identity") +
    scale_fill_manual(values = c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b")) +
    labs(title = "KNN Metrics Value", x = "Parameter", y = "Value") +
    theme_minimal() +
    theme(legend.position = "none")

  ggsave("results/KNNMetricsValue.png")
}

# Custom R2 function
R2_Score <- function(pred, actual) {
  ss_res <- sum((actual - pred)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  return(1 - ss_res / ss_tot)
}

# Example usage:
# process("Preprocessed_dataset.csv")
