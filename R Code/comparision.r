process <- function() {
  # Libraries
  library(ggplot2)
  library(readr)

  # Create data
  algorithms <- c("Naive Bayes", "SGD Classifier", "Voting Classifier")
  accuracy <- c(96.6819, 97.5973, 97.5114)
  colors <- c("#FF0000", "#1f77b4", "#008000")

  # Create data frame
  df <- data.frame(Algorithm = algorithms, Accuracy = accuracy, Color = colors)

  # Plot
  p <- ggplot(df, aes(x = Algorithm, y = Accuracy, fill = Algorithm)) +
    geom_bar(stat = "identity", show.legend = FALSE) +
    scale_fill_manual(values = df$Color) +
    labs(title = "Accuracy Comparison", x = "Algorithms", y = "Accuracy %") +
    theme_minimal()

  # Create results directory if it doesn't exist
  if (!dir.exists("results")) {
    dir.create("results")
  }

  # Save and show plot
  ggsave("results/comparision.png", plot = p)
  print(p)
  Sys.sleep(5)
}

# Run the process
process()
