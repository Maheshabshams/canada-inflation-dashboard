suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(forecast)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript arima_forecast.R <input_csv> <date_col> <target_col> <output_csv>")
}

input_csv <- args[1]
date_col  <- args[2]
target_col <- args[3]
output_csv <- args[4]

df <- read_csv(input_csv, show_col_types = FALSE) %>%
  arrange(.data[[date_col]])

y <- ts(df[[target_col]], frequency = 12)

fit <- auto.arima(y)
fc <- forecast(fit, h = 12)

out <- tibble(
  horizon = 1:12,
  forecast = as.numeric(fc$mean),
  lo95 = as.numeric(fc$lower[,2]),
  hi95 = as.numeric(fc$upper[,2])
)

write_csv(out, output_csv)
cat("Saved:", output_csv, "\n")
