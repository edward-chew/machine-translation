import os
import pandas as pd
from statistics import median, mean
import numpy as np
import scipy.stats as st
from sklearn.metrics import accuracy_score
import argparse
import csv

# The main routine
def main(dir_name: str, poly_label_col: str, true_label_col: str) -> None:
  accuracies = []
  noneu_accuracies = []

  # Bootstrap each language
  for rf in sorted(os.listdir(dir_name)):
    df = pd.read_csv(dir_name + "/" + rf)

    accuracies, noneu_accuracies = bootstrap(df, poly_label_col, true_label_col, sample_size=10000)

    # Save the accuracies to csv
    path = f"Bootstrapped/{dir_name.split('_')[0]}/AllTweets/"
    os.makedirs(path, exist_ok=True)  # Create the directory if it doesn't exist
    with open(path + rf.split('_')[0] + ".csv", "w") as wf:
      writer = csv.writer(wf)
      writer.writerow(accuracies)

    path = f"Bootstrapped/{dir_name.split('_')[0]}/NoNeuTweets/"
    os.makedirs(path, exist_ok=True)
    with open(path + rf.split('_')[0] + ".csv", "w") as wf:
      writer = csv.writer(wf)
      writer.writerow(noneu_accuracies)

  # Bootstrap all languages combined
  # Go through all the files, put all the files in a single dataframe
  df_list = []
  for rf in sorted(os.listdir(dir_name)):
    temp_df = pd.read_csv(dir_name + "/" + rf)
    df_list.append(temp_df)
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)

  accuracies, noneu_accuracies = bootstrap(combined_df, poly_label_col, true_label_col)

  # Save the accuracies to csv
  path = f"Bootstrapped/{dir_name.split('_')[0]}/AllTweets/"
  os.makedirs(path, exist_ok=True)
  with open(path + "Combined.csv", "w") as wf:
    writer = csv.writer(wf)
    writer.writerow(accuracies)

  path = f"Bootstrapped/{dir_name.split('_')[0]}/NoNeuTweets/"
  os.makedirs(path, exist_ok=True)
  with open(path + "Combined.csv", "w") as wf:
    writer = csv.writer(wf)
    writer.writerow(noneu_accuracies)
  

# Bootstrap a dataframe of sentiment labels
# If sample_size is not specified, the sample 
def bootstrap(df, poly_label_col: str, true_label_col: str, sample_size: int=0):
  accuracies = []
  noneu_accuracies = []

  for i in range(1000):
    if i % 10 == 0:
      print(i)

    if sample_size:
      sample_df = df.sample(n=sample_size, replace=True, random_state=i)  # Sample a specified sample size
    else:
      sample_df = df.sample(frac=1, replace=True, random_state=i)  # Sample the same number of rows as original df, with replacement
    accuracies.append(accuracy_score(sample_df[true_label_col], sample_df[poly_label_col]))
    
    # Calculate accuracies without neutrals
    noneu_sample_df = sample_df[(sample_df[poly_label_col] != "Neutral") & (sample_df[true_label_col] != "Neutral")]
    noneu_accuracies.append(accuracy_score(noneu_sample_df[true_label_col], noneu_sample_df[poly_label_col]))
  
  return accuracies, noneu_accuracies


if __name__ == "__main__":
  # parser = argparse.ArgumentParser()
  # parser.add_argument("file_directory", default="Twitter Dataset_CleanOutput", help="Name of the directory the tweet files are in")
  # parser.add_argument("poly_label_column_name", default="Tweet text_Clean_Label", help="Name of the column the Polyglot label is in")
  # parser.add_argument("true_label_column_name", default="SentLabel", help="Name of column the correct label is in")
  # args = parser.parse_args()

  # dir_name = args.file_directory
  # poly_label_col = args.poly_label_column_name
  # true_label_col = args.true_label_column_name

  dir_name = "Twitter Dataset New Languages_CleanOutput_10000Sample_EnglishToOriginal_PolyglotSentimentOutput"  # Name of the directory the tweet files are in
  poly_label_col = "ReverseTrans_Label"  # Name of the column the polyglot label is in
  true_label_col = "SentLabel"  # Name of column the correct label is in
  main(dir_name, poly_label_col, true_label_col)
