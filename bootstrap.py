import os
import pandas as pd
from statistics import median, mean
import numpy as np
import scipy.stats as st
from sentiment import calc_accuracy
import argparse

# The main routine
def main(dir_name: str, tweet_col: str, poly_label_col: str, true_label_col: str) -> None:
  # Go through all the files, put all the files in a signle dataframe
  df_list = []
  for rf in sorted(os.listdir(dir_name)):
    temp_df = pd.read_csv(dir_name + "/" + rf)
    df_list.append(temp_df)
    df = pd.concat(df_list, axis=0, ignore_index=True)

  accuracies = []
  noneu_accuracies = []

  for i in range(1000):
    if i % 10 == 0:
      print(i)

    # Sample the same number of rows as original df, with replacement
    sample_df = df.sample(frac=1, replace=True, random_state=i)
    accuracies.append(calc_accuracy(sample_df, poly_label_col, true_label_col))
    
    # Calculate accuracies without neutrals
    noneu_sample_df = sample_df[(sample_df[poly_label_col] != "Neutral") & (sample_df[true_label_col] != "Neutral")]
    noneu_accuracies.append(calc_accuracy(noneu_sample_df, poly_label_col, true_label_col))

  print("Median:", median(accuracies))
  print("Mean:", mean(accuracies))
  print("95 confidence:", st.t.interval(0.95, len(accuracies)-1, loc=np.mean(accuracies), scale=st.sem(accuracies)))

  print("Median no neu:", median(noneu_accuracies))
  print("Mean:", mean(noneu_accuracies))
  print("95 confidence no neu:",st.t.interval(0.95, len(noneu_accuracies)-1, loc=np.mean(noneu_accuracies), scale=st.sem(noneu_accuracies)))


if __name__ == "__main__":
  # parser = argparse.ArgumentParser()
  # parser.add_argument("file_directory", default="Twitter Dataset_CleanOutput", help="Name of the directory the tweet files are in")
  # parser.add_argument("tweet_column_name", default="Tweet text_Clean", help="Name of the column the tweet text is in")
  # parser.add_argument("poly_label_column_name", default="Tweet text_Clean_Label", help="Name of the column the Polyglot label is in")
  # parser.add_argument("true_label_column_name", default="SentLabel", help="Name of column the correct label is in")
  # args = parser.parse_args()

  # dir_name = args.file_directory
  # tweet_col = args.tweet_column_name
  # poly_label_col = args.poly_label_column_name
  # true_label_col = args.true_label_column_name

  dir_name = "EnglishToOriginalTweets_PolyglotSentimentOutput"  # Name of the directory the tweet files are in
  tweet_col = "ReverseTrans"  # Name of the column the tweet text is in
  poly_label_col = "ReverseTrans_Label"  # Name of the column the polyglot label is in
  true_label_col = "SentLabel"  # Name of column the correct label is in
  main(dir_name, tweet_col, poly_label_col, true_label_col)
