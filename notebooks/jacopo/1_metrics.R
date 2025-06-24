# by user
library(dplyr)
library(tibble)
library(quanteda)
library(quanteda.textstats) 


setwd("~/Desktop/decoding_language")

process_toks_return <- function(data) {
  lexical_complexity <- textstat_lexdiv(data, measure = c("TTR", "K"))
  
  n_token <- as.data.frame(ntoken(data)) |> tibble::rownames_to_column(var = "rowname")
  n_type <- as.data.frame(ntype(data)) |> tibble::rownames_to_column(var = "rowname")
  
  metrics_by_user <- lexical_complexity |>
    left_join(n_token, by = c("document" = "rowname")) |>
    left_join(n_type, by = c("document" = "rowname")) |>
    rename(
      "user_id" = "document",
      "n_token" = "ntoken(data)",
      "n_types" = "ntype(data)"
    )
  
  return(metrics_by_user)
}

# Carica i dati
cop26_toks <- readRDS("data/processed/by_user/toks_cop26_by_user.rds")
covid_toks <- readRDS("data/processed/by_user/toks_covid_by_user.rds")
ukraine_toks <- readRDS("data/processed/by_user/toks_ukraine_by_user.rds")

# Calcola le metriche per ciascun dataset
metrics_cop26 <- process_toks_return(cop26_toks) %>% mutate(dataset = "COP26")
metrics_covid <- process_toks_return(covid_toks) %>% mutate(dataset = "COVID-19")
metrics_ukraine <- process_toks_return(ukraine_toks) %>% mutate(dataset = "Ukraine")

# Unisci tutto in un unico dataframe
all_metrics <- bind_rows(metrics_cop26, metrics_covid, metrics_ukraine)

# Se vuoi, salva il dataframe completo
write.csv(all_metrics, "data/metrics/metrics_all_datasets_by_user.csv", row.names = FALSE)


# by tweet

library(dplyr)
library(tibble)


process_toks_return <- function(data) {
  lexical_complexity <- textstat_lexdiv(data, measure = c("TTR"))
  
  n_token <- as.data.frame(ntoken(data)) |> tibble::rownames_to_column(var = "rowname")
  n_type <- as.data.frame(ntype(data)) |> tibble::rownames_to_column(var = "rowname")
  
  metrics_by_user <- lexical_complexity |>
    left_join(n_token, by = c("document" = "rowname")) |>
    left_join(n_type, by = c("document" = "rowname")) |>
    rename(
      "user_id" = "document",
      "n_token" = "ntoken(data)",
      "n_types" = "ntype(data)"
    )
  
  return(metrics_by_user)
}

# Carica i dati
cop26_toks <- readRDS("data/processed/by_tweet/toks_cop26_by_tweet.rds")
covid_toks <- readRDS("data/processed/by_tweet/toks_covid_by_tweet.rds")
ukraine_toks <- readRDS("data/processed/by_tweet/toks_ukraine_by_tweet.rds")

# Calcola le metriche per ciascun dataset
metrics_cop26 <- process_toks_return(cop26_toks) %>% mutate(dataset = "COP26")
metrics_covid <- process_toks_return(covid_toks) %>% mutate(dataset = "COVID-19")
metrics_ukraine <- process_toks_return(ukraine_toks) %>% mutate(dataset = "Ukraine")

# Unisci tutto in un unico dataframe
all_metrics <- bind_rows(metrics_cop26, metrics_covid, metrics_ukraine)

# Se vuoi, salva il dataframe completo
write.csv(all_metrics, "data/metrics/metrics_all_datasets_by_tweet.csv", row.names = FALSE)
