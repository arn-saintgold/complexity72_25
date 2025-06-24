# Install arrow package if not already installed
if (!requireNamespace("arrow", quietly = TRUE)) {
  install.packages("arrow")
}

library(arrow)

# Read the parquet file
data <- read_parquet("~/Desktop/complexity72_25/data/processed/by_user/matrix_cop26_by_user.parquet")

data <- read_parquet("~/Desktop/complexity72_25/data/processed/clean_cop26.parquet")
data
