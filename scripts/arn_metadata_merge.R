library(data.table)
library(dplyr)

dataset_names <- c('cop26','covid','ukraine')

dataset_name <- 'cop26'

tweets <- arrow::read_parquet(paste0("./data/raw/",dataset_name,"_tweets_en.parquet"))
setDT(tweets)
topic_info <- fread(paste0("./data/processed/document_info_",dataset_name,"_tweets_en.csv"), select=c('Document','Topic','Name','Representative_Docs'))

# Merge tweets and topics
merged_df <- merge(tweets, topic_info, by.x = 'text', by.y = 'Document')

merged_df%>%names

# Load user info
usr_info <- fread('./data/raw/influencers_summary_cop26.csv')