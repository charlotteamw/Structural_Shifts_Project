# Load required libraries
library(tm)
library(wordcloud2)

# Load the data from CSV
data <- read.csv("wordcloud_keywords.csv", header = TRUE)

# Extract keywords as a character vector
keywords <- unlist(strsplit(as.character(data$Keywords), ", "))

# Convert keywords to lowercase
keywords <- tolower(keywords)

# Remove plural forms using stringr package
library(stringr)
keywords <- str_replace(keywords, "s$", "")

# Convert the modified keywords back to a data frame
keywords <- as.data.frame(keywords)

keywords$keywords <- ifelse(keywords$keywords == "stable isotope analysi", "stable isotope", keywords$keywords)

library(tidyverse)

words <- keywords %>%
  group_by(keywords) %>%
  summarize(count = n()) %>%
  ungroup() %>%
  mutate(count = as.numeric(count)) %>%  # Convert count to numeric
  arrange(desc(count)) %>%              # Sort in descending order of count
  slice_max(order_by = count, n = 30)
# Keep the top 50 keywords based on count

colnames(words) <- c("word", "freq")

# Create the word cloud using wordcloud2
wordcloud2(data=words, size = 0.1)

write.csv(words,file = "wordcloud_data.csv", row.names = FALSE)


           