# Load necessary libraries
library(tcltk)
library(tcltk2)
library(readr)
library(randomForest)
library(caret)
library(e1071)

# Define a list of colors for UI styling
bgcolor <- "#DAF7A6"
fgcolor <- "black"

# Define a global variable for file path
file_path <- ""

# Define function to browse and select file
browseFile <- function(entryWidget) {
  file_path <<- tk_choose.files()
  tclvalue(entryWidget) <- file_path
}

# Define process functions (you'll implement actual logic here)
process_LR <- function(file) {
  tkmessageBox(message = "Logistic Regression Completed", icon = "info")
}

process_RF <- function(file) {
  tkmessageBox(message = "Random Forest Completed", icon = "info")
}

process_DT <- function(file) {
  tkmessageBox(message = "Decision Tree Completed", icon = "info")
}

process_GB <- function(file) {
  tkmessageBox(message = "Gradient Boosting Completed", icon = "info")
}

process_LGBM <- function(file) {
  tkmessageBox(message = "LightGBM Completed", icon = "info")
}

process_KNN <- function(file) {
  tkmessageBox(message = "KNN Completed", icon = "info")
}

process_VC <- function(file) {
  tkmessageBox(message = "Voting Classifier Completed", icon = "info")
}

# Predict function using popup prompts
predictPCOS <- function() {
  Age <- as.numeric(tkgetOpenFile())
  weight <- as.numeric(tclvalue(tkentryDialog("Enter Weight (kg)")))
  height <- as.numeric(tclvalue(tkentryDialog("Enter Height (cm)")))
  BMI <- as.numeric(tclvalue(tkentryDialog("Enter BMI")))
  
  blood_group <- tclvalue(tkentryDialog("Enter Blood Group (O+ ve)"))
  blood_map <- list("A+ ve"=11, "A- ve"=12, "B+ ve"=13, "B- ve"=14, 
                    "O+ ve"=15, "O- ve"=16, "AB+ ve"=17, "AB- ve"=18)
  bgroup <- ifelse(blood_group %in% names(blood_map), blood_map[[blood_group]], NA)
  
  ct <- tclvalue(tkentryDialog("Cycle (Regular/Irregular)"))
  ct <- ifelse(ct == "Regular", 2, 4)

  prag <- ifelse(tclvalue(tkentryDialog("Pregnant? (Yes/No)")) == "Yes", 1, 0)

  # Add similar prompts for all other required inputs...

  # This is where you'd pass the collected inputs to your prediction model
  tkmessageBox(message = "Prediction complete (stub).", icon = "info")
}

# Create main window
win <- tktoplevel()
tkwm.title(win, "Diagnosis of PCOS using ML")
tkconfigure(win, bg=bgcolor)
tkwm.geometry(win, "1000x600")

# Title label
title_label <- tklabel(win, text="Diagnosis of PCOS using ML", font="Helvetica 20 bold", bg=bgcolor, fg=fgcolor)
tkgrid(title_label, columnspan=3, pady=20)

# Dataset label and entry
dataset_label <- tklabel(win, text="Dataset:", font="Helvetica 12", bg=bgcolor, fg=fgcolor)
dataset_entry <- tkentry(win, width=50)
tkgrid(dataset_label, dataset_entry)

# Browse button
browse_btn <- tkbutton(win, text="Browse", command=function() browseFile(dataset_entry))
tkgrid(browse_btn, pady=5)

# ML Buttons
tkgrid(tkbutton(win, text="Logistic Regression", command=function() process_LR(tclvalue(dataset_entry))), pady=5)
tkgrid(tkbutton(win, text="Random Forest", command=function() process_RF(tclvalue(dataset_entry))), pady=5)
tkgrid(tkbutton(win, text="Decision Tree", command=function() process_DT(tclvalue(dataset_entry))), pady=5)
tkgrid(tkbutton(win, text="Gradient Boosting", command=function() process_GB(tclvalue(dataset_entry))), pady=5)
tkgrid(tkbutton(win, text="LightGBM", command=function() process_LGBM(tclvalue(dataset_entry))), pady=5)
tkgrid(tkbutton(win, text="KNN", command=function() process_KNN(tclvalue(dataset_entry))), pady=5)
tkgrid(tkbutton(win, text="Voting Classifier", command=function() process_VC(tclvalue(dataset_entry))), pady=5)

# Prediction button
tkgrid(tkbutton(win, text="Predict PCOS", command=predictPCOS), pady=10)
