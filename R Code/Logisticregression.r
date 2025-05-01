library(data.table)
library(caret)
library(ggplot2)
library(Metrics)

process <- function(path) {
  # Read the data
  data <- fread(path)
  
  # Drop identifier columns
  data[, `Sl. No` := NULL]
  data[, `Patient File No.` := NULL]
  
  # Clean column names
  setnames(data, old = names(data), new = gsub("[^A-Za-z0-9_]", "", names(data)))
  
  # Extract features and labels
  y <- as.factor(data$PCOSYN)
  data[, PCOSYN := NULL]
  X <- data
  
  # Split data
  set.seed(42)
  train_idx <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[train_idx, ]
  y_train <- y[train_idx]
  X_test <- X[-train_idx, ]
  y_test <- y[-train_idx]
  
  # Train Logistic Regression
  model <- glm(y_train ~ ., data = cbind(X_train, y_train), family = "binomial")
  
  # Predict
  prob_pred <- predict(model, newdata = X_test, type = "response")
  y_pred <- ifelse(prob_pred > 0.5, 1, 0)
  
  # Evaluation metrics
  mse <- mse(as.numeric(y_test) - 1, y_pred)
  mae <- mae(as.numeric(y_test) - 1, y_pred)
  r2 <- R2_Score(as.numeric(y_test) - 1, y_pred)
  rmse <- rmse(as.numeric(y_test) - 1, y_pred)
  acc <- sum(y_pred == (as.numeric(y_test) - 1)) / length(y_test)
  
  # Print metrics
  cat("---------------------------------------------------------\n")
  cat(sprintf("MSE VALUE FOR Logistic Regression IS %f\n", mse))
  cat(sprintf("MAE VALUE FOR Logistic Regression IS %f\n", mae))
  cat(sprintf("R-SQUARED VALUE FOR Logistic Regression IS %f\n", r2))
  cat(sprintf("RMSE VALUE FOR Logistic Regression IS %f\n", rmse))
  cat(sprintf("ACCURACY VALUE FOR Logistic Regression IS %f\n", acc))
  cat("---------------------------------------------------------\n")
  
  # Save predictions
  pred_df <- data.table(ID = 1:length(y_pred), PredictedValue = y_pred)
  dir.create("results", showWarnings = FALSE)
  fwrite(pred_df, "results/resultLR.csv")
  
  # Save metrics
  metrics <- data.table(
    Parameter = c("MSE", "MAE", "R-SQUARED", "RMSE", "ACCURACY"),
    Value = c(mse, mae, r2, rmse, acc)
  )
  fwrite(metrics, "results/LRMetrics.csv")
  
  # Plot
  colors <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b")
  p <- ggplot(metrics, aes(x = Parameter, y = Value, fill = Parameter)) +
    geom_bar(stat = "identity", show.legend = FALSE) +
    scale_fill_manual(values = colors) +
    ggtitle("Logistic Regression Metrics Value") +
    theme_minimal()
  ggsave("results/LRMetricsValue.png", plot = p)
}

# Helper R² function
R2_Score <- function(true, pred) {
  ss_res <- sum((true - pred)^2)
  ss_tot <- sum((true - mean(true))^2)
  return(1 - ss_res / ss_tot)
}

# Uncomment to run
# process("Preprecesed_dataset.csv")
