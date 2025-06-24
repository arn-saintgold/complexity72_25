library(quanteda)
library(dplyr)
library(arrow)
library(readr)
library(jsonlite)

setwd("~/Desktop/complexity72_25")

# Define base path
base_path <- "~/Desktop/dedoding_language/data/"

# Parquet files
cop26_tweets   <- read_parquet(file.path(base_path, "raw/cop26_tweets_en.parquet"))
covid_tweets   <- read_parquet(file.path(base_path, "raw/covid_tweets_en.parquet"))
ukraine_tweets <- read_parquet(file.path(base_path, "raw/ukraine_tweets_en.parquet"))
ukraine_tweets <- ukraine_tweets %>%
  mutate(replied_id = na_if(replied_id, ""))
# CSV files
influencers_cop26 <- read_csv(file.path(base_path, "raw/influencers_summary_cop26.csv"))
influencers_covid <- read_csv(file.path(base_path, "raw/influencers_summary_covid.csv"))
influencers_ru_ukr <- read_csv(file.path(base_path, "raw/influencers_summary_ru_ukr.csv"))

# JSON file
speeches <- fromJSON(file.path(base_path, "raw/speeches.json"))


library(dplyr)
library(quanteda)
library(stopwords)
library(arrow)

save_tokens__by_user <- function(df,output_path_2) {
  # Controllo base utenti unici
  cat("Number of unique authors:", length(unique(df$author_id)), "\n")
  
  # Keep only original tweets (no replies)
  df <- df[is.na(df$replied_id), ]
  
  # Filter English tweets
  df <- df[df$lang == 'en', ]
  
  # Remove hashtags and mentions
  df$preprocessed_text <- gsub("#\\S+", "", df$text)
  df$preprocessed_text <- gsub("@\\S+", "", df$preprocessed_text)
  
  # Convert to lowercase
  df$preprocessed_text <- tolower(df$preprocessed_text)
  
  # Group by author and concatenate texts
  df_grouped <- df %>% 
    group_by(author_id) %>% 
    summarise(preprocessed_text = paste(preprocessed_text, collapse = " ")) %>% 
    ungroup()
  
  # Create corpus
  corpus <- corpus(df_grouped$preprocessed_text)
  docnames(corpus) <- df_grouped$author_id
  
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
  

  saveRDS(toks,output_path_2)
  cat("(B) Preprocessed data saved to:", output_path_2, "\n")
  return(toks)
}


toks_cop26_by_user=save_tokens__by_user(cop26_tweets,file.path(base_path,'processed/by_user/toks_cop26_by_user.rds'))
length(unique(cop26_tweets$author_id))
toks_covid_by_user=save_tokens__by_user(covid_tweets,file.path(base_path,'processed/by_user/toks_covid_by_user.rds'))
length(unique(covid_tweets$author_id))
toks_ukraine_by_user=save_tokens__by_user(ukraine_tweets,file.path(base_path,'processed/by_user/toks_ukraine_by_user.rds'))
length(unique(ukraine_tweets$author_id))



