process <- function(path) {
  library(ggplot2)
  library(gbm)
  library(readr)
  library(dplyr)
  library(caret)
  library(Metrics)

  # Create results directory if not exists
  if (!dir.exists("results")) dir.create("results")

  # Load and prepare data
  data <- read_csv(path)

  # Drop unnecessary columns
  data <- data %>% select(-`Sl. No`, -`Patient File No.`)

  # Set features and target
  X <- data %>% select(-`PCOS (Y/N)`)
  y <- data$`PCOS (Y/N)`

  # Train-test split (70% train, 30% test)
  set.seed(42)
  trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[trainIndex, ]
  X_test <- X[-trainIndex, ]
  y_train <- y[trainIndex]
  y_test <- y[-trainIndex]

  # Fit Gradient Boosting Model
  model <- gbm.fit(
    x = as.matrix(X_train), 
    y = y_train,
    distribution = "bernoulli",
    n.trees = 100,
    interaction.depth = 1,
    shrinkage = 1.0,
    n.minobsinnode = 10,
    verbose = FALSE
  )

  # Predictions (using all trees)
  y_pred_prob <- predict(model, as.matrix(X_test), n.trees = 100, type = "response")
  y_pred <- ifelse(y_pred_prob > 0.5, 1, 0)

  # Evaluation metrics
  acc <- sum(y_pred == y_test) / length(y_test)
  mse_val <- mse(y_test, y_pred)
  mae_val <- mae(y_test, y_pred)
  r2_val <- R2_Score(y_pred, y_test)
  rmse_val <- rmse(y_test, y_pred)

  cat("---------------------------------------------------------\n")
  cat(sprintf("MSE VALUE FOR Gradient Boost: %.6f\n", mse_val))
  cat(sprintf("MAE VALUE FOR Gradient Boost: %.6f\n", mae_val))
  cat(sprintf("R-SQUARED VALUE FOR Gradient Boost: %.6f\n", r2_val))
  cat(sprintf("RMSE VALUE FOR Gradient Boost: %.6f\n", rmse_val))
  cat(sprintf("ACCURACY VALUE FOR Gradient Boost: %.6f\n", acc))
  cat("---------------------------------------------------------\n")

  # Save predictions
  predictions_df <- data.frame(ID = 1:length(y_pred), Predicted_Value = y_pred)
  write_csv(predictions_df, "results/resultGB.csv")

  # Save metrics
  metrics_df <- data.frame(
    Parameter = c("MSE", "MAE", "R-SQUARED", "RMSE", "ACCURACY"),
    Value = c(mse_val, mae_val, r2_val, rmse_val, acc)
  )
  write_csv(metrics_df, "results/GBMetrics.csv")

  # Plot metrics
  ggplot(metrics_df, aes(x = Parameter, y = Value, fill = Parameter)) +
    geom_bar(stat = "identity") +
    scale_fill_manual(values = c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b")) +
    labs(title = "Gradient Boost Metrics Value", x = "Parameter", y = "Value") +
    theme_minimal() +
    theme(legend.position = "none")

  ggsave("results/GBMetricsValue.png")
}

# Custom R² function
R2_Score <- function(pred, actual) {
  ss_res <- sum((actual - pred)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  return(1 - ss_res / ss_tot)
}

# Example usage:
# process("Preprecesed_dataset.csv")
