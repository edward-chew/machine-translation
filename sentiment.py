from polyglot.text import Text
from polyglot.detect.base import logger as polyglot_logger
import csv
import os
import pandas as pd
import numpy as np
from lang_codes import lang_codes

# The main routine
def main(dir_name):
  # Silence error messages
  polyglot_logger.setLevel("ERROR")

  # Go through all the files in the specified directory
  for rf in sorted(os.listdir(dir_name)):
    df = pd.read_csv(dir_name + "/" + rf)

    language = rf.split("_")[0]

    df["PolyglotSentiment"] = df.apply(lambda row: get_polarity(row.CleanTweetText, language), axis = 1)
    df["PolyglotLabel"] = df.apply(lambda row: get_label(row.PolyglotSentiment), axis = 1)

    print(f"Accuracy of {language} is {calc_accuracy(df)}")
    count_labels(df)

    df.to_csv("PolyglotSentimentOutput/" + language + "_polyglot.csv", index = False)


# Calculate accuracy of Polyglot labels
def calc_accuracy(df) -> float:
  correct_labels = df.apply(lambda row: True if row.SentLabel == row.PolyglotLabel else False, axis = 1)
  return len(correct_labels[correct_labels == True].index) / len(df.index) 


# Generate labeling statistics
def count_labels(df):
  labels = df.apply(check_label, axis = 1)
  
  val_counts = labels.value_counts()

  tot_neg_labels = len(df[df["PolyglotLabel"] == "Negative"])
  print(f'  Total neg labels: {tot_neg_labels} ({tot_neg_labels / len(df.index) :.1%})')
  tot_neu_labels = len(df[df["PolyglotLabel"] == "Neutral"])
  print(f'  Total neu labels: {tot_neu_labels} ({tot_neu_labels / len(df.index) :.1%})')
  tot_pos_labels = len(df[df["PolyglotLabel"] == "Positive"])
  print(f'  Total pos labels: {tot_pos_labels} ({tot_pos_labels / len(df.index) :.1%})')

  tot_neg = len(df[df["SentLabel"] == "Negative"])
  print(f'  Total neg:               {tot_neg}')
  print(f'    Neg correct:           {val_counts["neg correct"]} ({val_counts["neg correct"] / tot_neg :.1%})')
  print(f'    Neg mislabeled as neu: {val_counts["neg mislabled as neu"]} ({val_counts["neg mislabled as neu"] / tot_neg :.1%})')
  print(f'    Neg mislabeled as pos: {val_counts["neg mislabled as pos"]} ({val_counts["neg mislabled as pos"] / tot_neg :.1%})')

  tot_neu = len(df[df["SentLabel"] == "Neutral"])
  print(f'  Total neu:               {tot_neu}')
  print(f'    Neu correct:           {val_counts["neu correct"]} ({val_counts["neu correct"] / tot_neu :.1%})')
  print(f'    Neu mislabeled as neg: {val_counts["neu mislabled as neg"]} ({val_counts["neu mislabled as neg"] / tot_neu :.1%})')
  print(f'    Neu mislabeled as pos: {val_counts["neu mislabled as pos"]} ({val_counts["neu mislabled as pos"] / tot_neu :.1%})')

  tot_pos = len(df[df["SentLabel"] == "Positive"])
  print(f'  Total pos:               {tot_pos}')
  print(f'    Pos correct:           {val_counts["pos correct"]} ({val_counts["pos correct"] / tot_pos :.1%})')
  print(f'    Pos mislabeled as neg: {val_counts["pos mislabled as neg"]} ({val_counts["pos mislabled as neg"] / tot_pos :.1%})')
  print(f'    Pos mislabeled as neu: {val_counts["pos mislabled as neu"]} ({val_counts["pos mislabled as neu"] / tot_pos :.1%})')


# Output a string indicating the correct/incorrect labeling for a row
def check_label(row):
  if row.SentLabel == "Negative" and row.PolyglotLabel == "Negative":
    return "neg correct"
  elif row.SentLabel == "Neutral" and row.PolyglotLabel == "Neutral":
    return "neu correct"
  elif row.SentLabel == "Positive" and row.PolyglotLabel == "Positive":
    return "pos correct"

  if row.SentLabel == "Negative" and row.PolyglotLabel == "Neutral":
    return "neg mislabled as neu"
  elif row.SentLabel == "Negative" and row.PolyglotLabel == "Positive":
    return "neg mislabled as pos"
  elif row.SentLabel == "Neutral" and row.PolyglotLabel == "Negative":
    return "neu mislabled as neg"
  elif row.SentLabel == "Neutral" and row.PolyglotLabel == "Positive":
    return "neu mislabled as pos"
  elif row.SentLabel == "Positive" and row.PolyglotLabel == "Neutral":
    return "pos mislabled as neu"
  elif row.SentLabel == "Positive" and row.PolyglotLabel == "Negative":
    return "pos mislabled as neg"


# Convert numerical polarity to text label
def get_label(polarity: float) -> str:
  if np.isnan(polarity):
    return ""
  
  sentiment = "Neutral"
  if polarity < 0:
    sentiment = "Negative"
  elif polarity > 0:
    sentiment = "Positive"
  
  return sentiment


# Get polyglot polarity
def get_polarity(tweet: str, lang: str) -> float:
  t = Text(str(tweet), hint_language_code = lang_codes[lang])
  assert(t.language.code == lang_codes[lang])

  # Sentiment detector fails when polarity is 0
  try:
    return t.polarity
  except ZeroDivisionError as e:
    return 0


if __name__ == "__main__":
  dir_name = "CleanTweets"
  main(dir_name)
