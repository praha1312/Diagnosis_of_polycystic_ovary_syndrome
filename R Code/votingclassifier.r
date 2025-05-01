library(randomForest)
library(e1071)
library(caret)
library(ggplot2)
library(Metrics)
library(dplyr)

process <- function(path) {
  # Load dataset
  data <- read.csv(path)
  X <- data %>% select(-c("PCOS..Y.N.", "Sl. No", "Patient.File.No."))
  y <- data$PCOS..Y.N.
  
  # Split the data into training and test sets
  set.seed(7)
  train_index <- createDataPartition(y, p = 0.9, list = FALSE)
  X_train <- X[train_index, ]
  y_train <- y[train_index]
  X_test <- X[-train_index, ]
  y_test <- y[-train_index]

  # Initialize models
  model1 <- glm(y_train ~ ., data = cbind(X_train, y_train), family = binomial())  # Logistic Regression
  model2 <- randomForest(X_train, y_train)  # Random Forest
  model3 <- svm(X_train, y_train, type = "C-classification", kernel = "linear")  # Support Vector Machine
  model4 <- svm(X_train, y_train, type = "C-classification", kernel = "linear", cost = 1)  # SGD Classifier

  # Combine models into an ensemble (VotingClassifier equivalent)
  models <- list(model1, model2, model3, model4)

  # Prediction from each model
  preds <- sapply(models, function(model) {
    if (inherits(model, "randomForest")) {
      predict(model, X_test, type = "response")
    } else {
      predict(model, X_test, type = "response")
    }
  })

  # Majority voting (if multiple classifiers)
  y_pred <- apply(preds, 1, function(x) {
    as.integer(names(sort(table(x), decreasing = TRUE))[1])
  })
  
  # Performance metrics
  mse <- mean((y_pred - y_test)^2)
  mae <- mean(abs(y_pred - y_test))
  r2 <- 1 - sum((y_pred - y_test)^2) / sum((mean(y_test) - y_test)^2)
  rms <- sqrt(mse)
  accuracy <- sum(y_pred == y_test) / length(y_test)

  print("---------------------------------------------------------")
  print(paste("MSE Value for Voting Classifier:", mse))
  print(paste("MAE Value for Voting Classifier:", mae))
  print(paste("R-squared Value for Voting Classifier:", r2))
  print(paste("RMSE Value for Voting Classifier:", rms))
  print(paste("Accuracy Value for Voting Classifier:", accuracy))
  print("---------------------------------------------------------")

  # Save metrics to a CSV file
  metrics <- data.frame(
    Parameter = c("MSE", "MAE", "R-SQUARED", "RMSE", "ACCURACY"),
    Value = c(mse, mae, r2, rms, accuracy)
  )
  write.csv(metrics, 'results/VCMetrics.csv', row.names = FALSE)

  # Plot metrics
  ggplot(metrics, aes(x = Parameter, y = Value, fill = Parameter)) +
    geom_bar(stat = "identity") +
    labs(title = "VotingClassifier Metrics Value", x = "Parameter", y = "Value") +
    scale_fill_manual(values = c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b")) +
    ggsave('results/VCMetricsValue.png')
}

# Example usage
# process("your_dataset.csv")
