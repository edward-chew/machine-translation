from clean import main as clean
from sentiment import main as sentiment
import sys

if len(sys.argv) == 3:
  dir_name = sys.argv[1]  # Name of the directory the tweet files are in
  tweet_col = sys.argv[2]  # Name of the column the tweet text is in
else:
  print("Usage: python translation_project.py <file_directory> <tweet_column_name>")
  exit()

clean(dir_name, tweet_col)

clean_dir_name = dir_name + "_CleanOutput"  # Name of the directory the tweet files are in
clean_tweet_col = tweet_col + "_Clean"  # Name of the column the tweet text is in
true_label_col = "SentLabel"  # Name of column the correct label is in
sentiment(clean_dir_name, clean_tweet_col, true_label_col)
