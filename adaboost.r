# Load necessary libraries
library(caret)
library(randomForest)
library(readr)
library(ggplot2)

process <- function(path) {
  # Read the dataset
  data <- read_csv(path)
  
  # Drop unnecessary columns
  X <- data[, !names(data) %in% c("PCOS (Y/N)", "Sl. No", "Patient File No.")]
  y <- data$`PCOS (Y/N)`
  
  # Split the data into training and testing sets
  set.seed(123) # For reproducibility
  trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[trainIndex, ]
  X_test <- X[-trainIndex, ]
  y_train <- y[trainIndex]
  y_test <- y[-trainIndex]
  
  # Fit AdaBoost model using gbm (Gradient Boosting Machine)
  library(gbm)
  set.seed(1)
  rfc <- gbm.fit(
    X_train,
    y_train,
    distribution = "bernoulli",
    n.trees = 100,
    interaction.depth = 1,
    shrinkage = 1,
    bag.fraction = 0.5,
    train.fraction = 1,
    verbose = FALSE
  )
  
  # Make predictions
  y_pred <- predict(rfc, X_test, n.trees = 100, type = "response")
  y_pred <- ifelse(y_pred > 0.5, 1, 0) # Convert probabilities to binary predictions
  
  # Calculate accuracy
  accuracy <- mean(y_pred == y_test)
  print(paste("Accuracy:", accuracy))
  
  # Save predictions to a CSV file
  result_df <- data.frame(ID = 1:length(y_pred), Predicted_Value = y_pred)
  write.csv(result_df, "results/resultadaboost.csv", row.names = FALSE)
  
  # Calculate evaluation metrics
  mse <- mean((y_test - y_pred)^2)
  mae <- mean(abs(y_test - y_pred))
  r2 <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)
  rmse <- sqrt(mse)
  
  cat("---------------------------------------------------------\n")
  cat(paste("MSE VALUE FOR adaboost IS", mse, "\n"))
  cat(paste("MAE VALUE FOR adaboost IS", mae, "\n"))
  cat(paste("R-SQUARED VALUE FOR adaboost IS", r2, "\n"))
  cat(paste("RMSE VALUE FOR adaboost IS", rmse, "\n"))
  cat(paste("ACCURACY VALUE adaboost IS", accuracy, "\n"))
  cat("---------------------------------------------------------\n")
  
  # Save metrics to a CSV file
  metrics_df <- data.frame(
    Parameter = c("MSE", "MAE", "R-SQUARED", "RMSE", "ACCURACY"),
    Value = c(mse, mae, r2, rmse, accuracy)
  )
  write.csv(metrics_df, "results/adaboostMetrics.csv", row.names = FALSE)
  
  # Plot the metrics
  ggplot(metrics_df, aes(x = Parameter, y = Value, fill = Parameter)) +
    geom_bar(stat = "identity") +
    labs(title = "AdaBoost Metrics Value", x = "Parameter", y = "Value") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave("results/adaboostMetricsValue.png")
}

# Uncomment the line below to run the function with your dataset path
process("dataset.csv")