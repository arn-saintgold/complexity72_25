library(quanteda)
library(dplyr)
library(arrow)
library(readr)
library(jsonlite)
setwd("~/Desktop/complexity72_25")

# Parquet files
cop26_tweets <- read_parquet("data/raw/cop26_tweets_en.parquet")
covid_tweets <- read_parquet("data/raw/covid_tweets_en.parquet")
ukraine_tweets <- read_parquet("data/raw/ukraine_tweets_en.parquet")
ukraine_tweets <- ukraine_tweets %>%
  mutate(replied_id = na_if(replied_id, ""))


# CSV files
influencers_cop26 <- read_csv("data/raw/influencers_summary_cop26.csv")
influencers_covid <- read_csv("data/raw/influencers_summary_covid.csv")
influencers_ru_ukr <- read_csv("data/raw/influencers_summary_ru_ukr.csv")

# JSON file
speeches <- fromJSON("data/raw/speeches.json")





save_tokens_by_post <- function(df,output_path,output_path_2) {
  
  # Keep only original tweets (no replies)
  df <- df[is.na(df$replied_id), ]
  
  # Filter English tweets
  df <- df[df$lang == 'en', ]
  
  # Remove hashtags and mentions
  df$preprocessed_text <- gsub("#\\S+", "", df$text)
  df$preprocessed_text <- gsub("@\\S+", "", df$preprocessed_text)
  
  # Convert to lowercase
  df$preprocessed_text <- tolower(df$preprocessed_text)
  
  write_parquet(df,output_path_2)
  # Create corpus
  corpus <- corpus(df$preprocessed_text)
  docnames(corpus) <- df$id
  
  # Tokenization and preprocessing
  toks <- tokens(corpus,
                 remove_punct = TRUE,
                 remove_symbols = TRUE,
                 remove_numbers = TRUE,
                 remove_url = TRUE,
                 remove_separators = TRUE,
                 split_hyphens = TRUE) |>
    tokens_remove(stopwords("en")) |>
    tokens_wordstem(language = "en")
  
  
  saveRDS(toks,output_path)
  cat("(B) Preprocessed data saved to:", output_path, "\n")
  return(toks)
}


toks_cop26_by_tweet=save_tokens_by_post(cop26_tweets,"~/Desktop/complexity72_25/data/processed/by_tweet/toks_cop26_by_tweet.rds","~/Desktop/complexity72_25/data/processed/clean_cop26.parquet")
toks_covid_by_tweet=save_tokens_by_post(covid_tweets,"~/Desktop/complexity72_25/data/processed/by_tweet/toks_covid_by_tweet.rds","~/Desktop/complexity72_25/data/processed/clean_covid.parquet")
toks_ukraine_by_user=save_tokens_by_post(ukraine_tweets,"~/Desktop/complexity72_25/data/processed/by_tweet/toks_ukraine_by_tweet.rds","~/Desktop/complexity72_25/data/processed/clean_ukraine.parquet")


