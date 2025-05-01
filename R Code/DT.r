process <- function(path) {
  # Libraries
  library(rpart)
  library(caret)
  library(Metrics)
  library(ggplot2)
  library(dplyr)
  
  # Read data
  data <- read.csv(path)
  
  # Drop columns: "Sl. No", "Patient File No.", and the target column from X
  X <- data %>% select(-c("PCOS..Y.N.", "Sl..No", "Patient.File.No."))
  y <- data$PCOS..Y.N.
  
  # Split into train and test sets
  set.seed(100)
  index <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[index, ]
  X_test <- X[-index, ]
  y_train <- y[index]
  y_test <- y[-index]
  
  # Train Decision Tree
  df_train <- cbind(X_train, PCOS = y_train)
  model <- rpart(PCOS ~ ., data = df_train, method = "class", 
                 control = rpart.control(maxdepth = 3, minbucket = 5))
  
  # Predict
  y_pred <- predict(model, X_test, type = "class")
  accuracy <- sum(y_pred == y_test) / length(y_test)
  print(accuracy)
  
  # Save predictions
  if (!dir.exists("results")) dir.create("results")
  results_df <- data.frame(ID = 1:length(y_pred), Predicted_Value = y_pred)
  write.csv(results_df, "results/resultDT.csv", row.names = FALSE)
  
  # Metrics
  y_test_num <- as.numeric(as.character(y_test))
  y_pred_num <- as.numeric(as.character(y_pred))
  mse <- mse(y_test_num, y_pred_num)
  mae <- mae(y_test_num, y_pred_num)
  r2 <- R2(y_pred_num, y_test_num)
  rmse_val <- rmse(y_test_num, y_pred_num)
  ac <- accuracy
  
  cat("---------------------------------------------------------\n")
  cat(sprintf("MSE VALUE FOR Decision Tree IS %f\n", mse))
  cat(sprintf("MAE VALUE FOR Decision Tree IS %f\n", mae))
  cat(sprintf("R-SQUARED VALUE FOR Decision Tree IS %f\n", r2))
  cat(sprintf("RMSE VALUE FOR Decision Tree IS %f\n", rmse_val))
  cat(sprintf("ACCURACY VALUE FOR Decision Tree IS %f\n", ac))
  cat("---------------------------------------------------------\n")
  
  # Save metrics
  metrics_df <- data.frame(
    Parameter = c("MSE", "MAE", "R-SQUARED", "RMSE", "ACCURACY"),
    Value = c(mse, mae, r2, rmse_val, ac)
  )
  write.csv(metrics_df, "results/DTMetrics.csv", row.names = FALSE)
  
  # Plot
  colors <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b")
  p <- ggplot(metrics_df, aes(x = Parameter, y = Value, fill = Parameter)) +
    geom_bar(stat = "identity", show.legend = FALSE) +
    scale_fill_manual(values = colors) +
    labs(title = "Decision Tree Metrics Value", x = "Parameter", y = "Value") +
    theme_minimal()
  
  ggsave("results/DTMetricsValue.png", plot = p)
  print(p)
  Sys.sleep(5)
}

# Example usage (uncomment to run)
process("dataset.csv")
