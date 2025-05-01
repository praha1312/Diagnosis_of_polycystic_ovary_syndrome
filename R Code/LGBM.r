library(data.table)
library(lightgbm)
library(caret)
library(Metrics)
library(ggplot2)
library(stringr)

process <- function(path) {
  # Load the dataset
  data <- fread(path)
  
  # Drop identifier columns
  data[, `Sl. No` := NULL]
  data[, `Patient File No.` := NULL]

  # Clean column names (remove special characters)
  setnames(data, old = names(data), new = str_replace_all(names(data), "[^A-Za-z0-9_]", ""))
  
  # Extract features and labels
  y <- as.factor(data$PCOSYN)
  data[, PCOSYN := NULL]  # Remove label from features
  X <- data
  
  # Split dataset
  set.seed(123)
  train_idx <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[train_idx, ]
  y_train <- y[train_idx]
  X_test <- X[-train_idx, ]
  y_test <- y[-train_idx]
  
  # Convert data to lgb.Dataset format
  dtrain <- lgb.Dataset(data = as.matrix(X_train), label = as.numeric(y_train) - 1)
  
  # Train LightGBM classifier
  params <- list(objective = "binary", metric = "binary_logloss")
  model <- lgb.train(params, dtrain, nrounds = 100)
  
  # Predict
  preds_prob <- predict(model, as.matrix(X_test))
  preds <- ifelse(preds_prob > 0.5, 1, 0)
  
  # Metrics
  mse <- mse(as.numeric(y_test) - 1, preds)
  mae <- mae(as.numeric(y_test) - 1, preds)
  r2 <- R2_Score(as.numeric(y_test) - 1, preds)
  rmse <- rmse(as.numeric(y_test) - 1, preds)
  acc <- sum(preds == (as.numeric(y_test) - 1)) / length(y_test)
  
  # Print Metrics
  cat("---------------------------------------------------------\n")
  cat(sprintf("MSE VALUE FOR LGBM IS %f\n", mse))
  cat(sprintf("MAE VALUE FOR LGBM IS %f\n", mae))
  cat(sprintf("R-SQUARED VALUE FOR LGBM IS %f\n", r2))
  cat(sprintf("RMSE VALUE FOR LGBM IS %f\n", rmse))
  cat(sprintf("ACCURACY VALUE FOR LGBM IS %f\n", acc))
  cat("---------------------------------------------------------\n")
  
  # Save predicted values
  results <- data.table(ID = 1:length(preds), PredictedValue = preds)
  dir.create("results", showWarnings = FALSE)
  fwrite(results, "results/resultLGBM.csv")
  
  # Save metrics
  metrics <- data.table(
    Parameter = c("MSE", "MAE", "R-SQUARED", "RMSE", "ACCURACY"),
    Value = c(mse, mae, r2, rmse, acc)
  )
  fwrite(metrics, "results/LGBMMetrics.csv")
  
  # Plot metrics
  p <- ggplot(metrics, aes(x = Parameter, y = Value, fill = Parameter)) +
    geom_bar(stat = "identity") +
    ggtitle("LGBM Metrics Value") +
    xlab("Parameter") + ylab("Value") +
    theme_minimal()
  ggsave("results/LGBMMetricsValue.png", plot = p)
}

# Helper R2 Score function
R2_Score <- function(true, predicted) {
  ss_res <- sum((true - predicted)^2)
  ss_tot <- sum((true - mean(true))^2)
  return (1 - ss_res / ss_tot)
}

# Uncomment to run
# process("Preprecesed_dataset.csv")
