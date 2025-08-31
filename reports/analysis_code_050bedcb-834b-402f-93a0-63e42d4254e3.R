# DataScribe Analysis Code (R)
# Generated on: 2025-08-31 23:03:06
# Dataset: winequality-red.csv

# Load required libraries
library(ggplot2)
library(dplyr)
library(corrplot)
library(gridExtra)

# Load your dataset
# df <- read.csv('your_dataset.csv')  # Replace with your file path

# For demonstration, we'll create a sample dataframe
# Replace this with your actual data loading code
set.seed(42)
n_samples <- 1000
df <- data.frame(
    feature_1 = rnorm(n_samples, 0, 1),
    feature_2 = rnorm(n_samples, 0, 1),
    feature_3 = sample(c('A', 'B', 'C'), n_samples, replace = TRUE),
    target = sample(c(0, 1), n_samples, replace = TRUE)
)

# Basic dataset info
cat("Dataset Shape:", nrow(df), "rows x", ncol(df), "columns\n")
cat("Memory Usage:", object.size(df) / 1024^2, "MB\n")
cat("\nData Types:\n")
str(df)

# Data Quality Assessment
cat("\n=== Data Quality Assessment ===\n")
cat("Missing Values:\n")
print(colSums(is.na(df)))

cat("\nDuplicate Rows:", sum(duplicated(df)), "\n")

# Statistical Summary
cat("\n=== Statistical Summary ===\n")
print(summary(df))

# Column Analysis
numerical_cols <- sapply(df, is.numeric)
categorical_cols <- sapply(df, is.character) | sapply(df, is.factor)

cat("\nNumerical Columns:", sum(numerical_cols), "\n")
cat("Categorical Columns:", sum(categorical_cols), "\n")

# Visualizations
if(sum(numerical_cols) > 0) {
    # Correlation Matrix
    cor_matrix <- cor(df[, numerical_cols], use="complete.obs")
    corrplot(cor_matrix, method="color", type="upper", 
             order="hclust", tl.cex=0.7, tl.col="black")
    
    # Distribution plots for numerical columns
    num_cols <- names(df)[numerical_cols]
    for(col in head(num_cols, 5)) {  # Limit to first 5 columns
        p1 <- ggplot(df, aes_string(x=col)) +
               geom_histogram(fill="skyblue", alpha=0.7, bins=30) +
               labs(title=paste("Distribution of", col)) +
               theme_minimal()
        
        p2 <- ggplot(df, aes_string(y=col)) +
               geom_boxplot(fill="lightgreen", alpha=0.7) +
               labs(title=paste("Box Plot of", col)) +
               theme_minimal()
        
        grid.arrange(p1, p2, ncol=2)
    }
}

if(sum(categorical_cols) > 0) {
    # Bar plots for categorical columns
    cat_cols <- names(df)[categorical_cols]
    for(col in head(cat_cols, 3)) {  # Limit to first 3 columns
        p <- ggplot(df, aes_string(x=col)) +
             geom_bar(fill="steelblue", alpha=0.7) +
             labs(title=paste("Distribution of", col)) +
             theme_minimal() +
             theme(axis.text.x = element_text(angle = 45, hjust = 1))
        print(p)
    }
}

# Data Cleaning Recommendations
cat("\n=== Data Cleaning Recommendations ===\n")
for(col in names(df)) {
    missing_pct <- sum(is.na(df[[col]])) / nrow(df) * 100
    if(missing_pct > 0) {
        cat(col, ":", round(missing_pct, 1), "% missing values\n")
    }
    
    if(is.character(df[[col]]) || is.factor(df[[col]])) {
        unique_vals <- length(unique(df[[col]]))
        if(unique_vals == 1) {
            cat(col, ": Constant column (only one unique value)\n")
        } else if(unique_vals == nrow(df)) {
            cat(col, ": High cardinality (all values unique)\n")
        }
    }
}

cat("\n=== Analysis Complete ===\n")
cat("This code provides a basic EDA framework in R.\n")
cat("Customize it based on your specific dataset and analysis goals.\n")
