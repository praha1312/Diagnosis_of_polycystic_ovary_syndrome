# Load necessary libraries
library(randomForest)
library(caret)
library(ggplot2)
library(readr)

# Function to process the dataset and train a Random Forest model
process <- function(path) {
  # Read the dataset
  data <- read_csv(path)
  
  # Drop irrelevant columns
  data$`Sl. No` <- NULL
  data$`Patient File No.` <- NULL

  # Convert target variable to factor
  data$`PCOS (Y/N)` <- as.factor(data$`PCOS (Y/N)`)

  # Split data into features and target
  X <- data[, !names(data) %in% c("PCOS (Y/N)")]
  y <- data$`PCOS (Y/N)`

  # Train-test split
  set.seed(123)
  trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)
  trainData <- data[trainIndex, ]
  testData <- data[-trainIndex, ]

  # Train Random Forest model
  rf_model <- randomForest(`PCOS (Y/N)` ~ ., data = trainData)

  # Predict on test data
  predictions <- predict(rf_model, testData)

  # Save predictions
  results <- data.frame(ID = 1:length(predictions), Predicted = predictions)
  dir.create("results", showWarnings = FALSE)
  write_csv(results, "results/resultRF.csv")

  # Evaluation Metrics
  confusion <- confusionMatrix(predictions, testData$`PCOS (Y/N)`)
  accuracy <- confusion$overall['Accuracy']
  mse <- mean((as.numeric(predictions) - as.numeric(testData$`PCOS (Y/N)`))^2)
  mae <- mean(abs(as.numeric(predictions) - as.numeric(testData$`PCOS (Y/N)`)))
  rmse <- sqrt(mse)
  r2 <- R2(as.numeric(predictions), as.numeric(testData$`PCOS (Y/N)`))

  # Print Metrics
  cat("---------------------------------------------------------\n")
  cat(sprintf("MSE VALUE FOR Random Forest IS %f\n", mse))
  cat(sprintf("MAE VALUE FOR Random Forest IS %f\n", mae))
  cat(sprintf("R-SQUARED VALUE FOR Random Forest IS %f\n", r2))
  cat(sprintf("RMSE VALUE FOR Random Forest IS %f\n", rmse))
  cat(sprintf("ACCURACY VALUE FOR Random Forest IS %f\n", accuracy))
  cat("---------------------------------------------------------\n")

  # Save metrics
  metrics <- data.frame(
    Parameter = c("MSE", "MAE", "R-SQUARED", "RMSE", "ACCURACY"),
    Value = c(mse, mae, r2, rmse, accuracy)
  )
  write_csv(metrics, "results/RFMetrics.csv")

  # Plot metrics
  ggplot(metrics, aes(x = Parameter, y = Value, fill = Parameter)) +
    geom_bar(stat = "identity") +
    labs(title = "Random Forest Metrics Value", x = "Parameter", y = "Value") +
    theme_minimal() +
    theme(legend.position = "none") +
    scale_fill_brewer(palette = "Set2")
  
  ggsave("results/RFMetricsValue.png")
}

# Example usage:
# process("your_dataset.csv")
