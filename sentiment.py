from polyglot.text import Text
from polyglot.detect.base import logger as polyglot_logger
import os
import pandas as pd
import numpy as np
from nptyping import Float64
from lang_codes import lang_codes
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import argparse

# The main routine
def main(dir_name: str, tweet_col: str, true_label_col: str) -> None:
  # Silence error messages
  polyglot_logger.setLevel("ERROR")

  print("--- Results ---")

  accuracies = { "overall" : {},
                 "no_neu"  : {} }

  # Go through all the files in the specified directory
  for rf in sorted(os.listdir(dir_name)):
    df = pd.read_csv(dir_name + "/" + rf)

    language = rf.split(".")[0]

    # Get numerical sentiment
    poly_sentiment_col = f"{tweet_col}_Sentiment"
    df[poly_sentiment_col] = df.apply(lambda row: get_polarity(row.get(tweet_col), language), axis = 1)
    # Convert numerical sentiment to text labels
    poly_label_col = f"{tweet_col}_Label"
    df[poly_label_col] = df.apply(lambda row: get_label(row.get(poly_sentiment_col)), axis = 1)

    # Print accuracies / stats
    accuracy = accuracy_score(df[true_label_col], df[poly_label_col])
    accuracies["overall"][language] = accuracy
    print(f"Accuracy of {language} is              {accuracy}")

    noneu_df = df[(df[poly_label_col] != "Neutral") & (df[true_label_col] != "Neutral")]
    accuracy = accuracy_score(noneu_df[true_label_col], noneu_df[poly_label_col])
    accuracies["no_neu"][language] = accuracy
    print(f"Accuracy of {language} w/o neutrals is {accuracy}")

    # count_labels(df, poly_label_col, true_label_col)

    # Confusion Matrix
    print_confusion_matrix(df, poly_label_col, true_label_col)

    # Output to file
    output_dir_name = dir_name + "_PolyglotSentimentOutput"
    if not os.path.exists(output_dir_name):
      os.makedirs(output_dir_name)
    df.to_csv(output_dir_name + "/" + language + ".csv", index = False)

  print(accuracies)


# Calculate accuracy of Polyglot labels
def calc_accuracy(df, poly_label_col: str, true_label_col: str) -> float:
  correct_labels = df.apply(lambda row: True if row.get(true_label_col) == row.get(poly_label_col) else False, axis = 1)
  return len(correct_labels[correct_labels == True].index) / len(df.index)


# Generate confusion matrix
def print_confusion_matrix(df, poly_label_col: str, true_label_col: str) -> None:
  y_true = df[true_label_col].tolist()
  y_pred = df[poly_label_col].tolist()

  l = ["Positive", "Negative", "Neutral"]
  cm = confusion_matrix(y_true, y_pred, labels=l)
  print("  Confusion matrix: \n", cm)


# Generate labeling statistics
def count_labels(df, poly_label_col: str, true_label_col: str):
  labels = df.apply(lambda row: check_label(row, poly_label_col, true_label_col), axis = 1)
  
  val_counts = labels.value_counts()

  tot_neg_labels = len(df[df[poly_label_col] == "Negative"])
  print(f'  Total neg labels:        {tot_neg_labels} ({tot_neg_labels / len(df.index) :.1%})')
  tot_neu_labels = len(df[df[poly_label_col] == "Neutral"])
  print(f'  Total neu labels:        {tot_neu_labels} ({tot_neu_labels / len(df.index) :.1%})')
  tot_pos_labels = len(df[df[poly_label_col] == "Positive"])
  print(f'  Total pos labels:        {tot_pos_labels} ({tot_pos_labels / len(df.index) :.1%})')

  tot_neg = len(df[df[true_label_col] == "Negative"])
  print(f'  Total neg:               {tot_neg}')
  print(f'    Neg correct:           {val_counts["neg correct"]} ({val_counts["neg correct"] / tot_neg :.1%})')
  print(f'    Neg mislabeled as neu: {val_counts["neg mislabled as neu"]} ({val_counts["neg mislabled as neu"] / tot_neg :.1%})')
  print(f'    Neg mislabeled as pos: {val_counts["neg mislabled as pos"]} ({val_counts["neg mislabled as pos"] / tot_neg :.1%})')

  tot_neu = len(df[df[true_label_col] == "Neutral"])
  print(f'  Total neu:               {tot_neu}')
  print(f'    Neu correct:           {val_counts["neu correct"]} ({val_counts["neu correct"] / tot_neu :.1%})')
  print(f'    Neu mislabeled as neg: {val_counts["neu mislabled as neg"]} ({val_counts["neu mislabled as neg"] / tot_neu :.1%})')
  print(f'    Neu mislabeled as pos: {val_counts["neu mislabled as pos"]} ({val_counts["neu mislabled as pos"] / tot_neu :.1%})')

  tot_pos = len(df[df[true_label_col] == "Positive"])
  print(f'  Total pos:               {tot_pos}')
  print(f'    Pos correct:           {val_counts["pos correct"]} ({val_counts["pos correct"] / tot_pos :.1%})')
  print(f'    Pos mislabeled as neg: {val_counts["pos mislabled as neg"]} ({val_counts["pos mislabled as neg"] / tot_pos :.1%})')
  print(f'    Pos mislabeled as neu: {val_counts["pos mislabled as neu"]} ({val_counts["pos mislabled as neu"] / tot_pos :.1%})')


# Output a string indicating the correct/incorrect labeling for a row
def check_label(row, poly_label_col: str, true_label_col: str):
  if row.get(true_label_col) == "Negative" and row.get(poly_label_col) == "Negative":
    return "neg correct"
  elif row.get(true_label_col) == "Neutral" and row.get(poly_label_col) == "Neutral":
    return "neu correct"
  elif row.get(true_label_col) == "Positive" and row.get(poly_label_col) == "Positive":
    return "pos correct"

  if row.get(true_label_col)== "Negative" and row.get(poly_label_col) == "Neutral":
    return "neg mislabled as neu"
  elif row.get(true_label_col) == "Negative" and row.get(poly_label_col) == "Positive":
    return "neg mislabled as pos"
  elif row.get(true_label_col) == "Neutral" and row.get(poly_label_col) == "Negative":
    return "neu mislabled as neg"
  elif row.get(true_label_col) == "Neutral" and row.get(poly_label_col) == "Positive":
    return "neu mislabled as pos"
  elif row.get(true_label_col) == "Positive" and row.get(poly_label_col) == "Neutral":
    return "pos mislabled as neu"
  elif row.get(true_label_col) == "Positive" and row.get(poly_label_col) == "Negative":
    return "pos mislabled as neg"


# Convert numerical polarity to text label
def get_label(polarity: Float64) -> str:
  if polarity is None:
    return ""
  
  sentiment = "Neutral"
  if polarity < 0:
    sentiment = "Negative"
  elif polarity > 0:
    sentiment = "Positive"
  
  return sentiment


# Get polyglot polarity
def get_polarity(tweet: str, lang: str) -> Float64:
  t = Text(str(tweet), hint_language_code = lang_codes[lang])
  assert(t.language.code == lang_codes[lang])

  # Sentiment detector fails when polarity is 0
  try:
    return t.polarity
  except ZeroDivisionError as e:
    return 0


if __name__ == "__main__":
  # parser = argparse.ArgumentParser()
  # parser.add_argument("file_directory", default="Twitter Dataset_CleanOutput", help="Name of the directory the tweet files are in")
  # parser.add_argument("tweet_column_name", default="Tweet text_Clean", help="Name of the column the tweet text is in")
  # parser.add_argument("true_label_column_name", default="SentLabel", help="Name of column the correct label is in")
  # args = parser.parse_args()

  # dir_name = args.file_directory
  # tweet_col = args.tweet_column_name
  # true_label_col = args.true_label_column_name

  dir_name = "Twitter Dataset New Languages_CleanOutput_10000Sample_EnglishToOriginal_PolyglotSentimentOutput_PolyglotSentimentOutput"
  tweet_col = "ReverseTrans"
  true_label_col = "SentLabel"

  main(dir_name, tweet_col, true_label_col)
