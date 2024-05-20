import os
import re
import pandas as pd
import argparse


def main(dir_name: str, tweet_col: str, sent_label_col: str) -> None:
  print("--- Cleaning ---")
  # Go through all the files in the specified directory
  for rf in sorted(os.listdir(dir_name)):
    print(rf)

    df = pd.read_csv(dir_name + "/" + rf, on_bad_lines="skip")
    
    language = rf.split("_")[0]

    # Rename or generate new columns so it is consistent between two datasets
    if tweet_col != "Tweet text":
      df.rename(columns={tweet_col: "Tweet text"}, inplace=True)
    
    if sent_label_col != "SentLabel":
      df["SentLabel"] = df.apply(lambda row: convert_sentiment_labels(row.get(sent_label_col)), axis = 1)

    # Remove tweets that are empty
    df = df[df["Tweet text"].notna()]

    # Clean tweet text
    df[f"Tweet text_Clean"] = df.apply(lambda row: clean_text(row.get("Tweet text")), axis = 1)

    # Remove tweets that are empty after cleaning
    df = df[df["Tweet text"].notna()]

    # Output to file
    output_dir_name = dir_name + "_CleanOutput"
    if not os.path.exists(output_dir_name):
      os.makedirs(output_dir_name)
    df.to_csv(output_dir_name + "/" + language + ".csv", index = False)


def clean_text(text: str) -> str:
  # remove twitter retweet handles (RT @xxx:)
  text = re.sub("RT @[\w]*:", "", text)
  # remove twitter handles (@xxx)
  text = re.sub("@[\w]*", "", text)
  # remove URL links (httpxxx)
  text = re.sub("https?://[A-Za-z0-9./]*", "", text)
  # remove numbers
  text = re.sub("[0-9]", " ", text)
  # make all characters lowercase
  text = text.lower()

  return text


def convert_sentiment_labels(val: int) -> str:
  if val == -1:
    return "Negative"
  elif val == 0:
    return "Neutral"
  else:
    return "Positive"


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file_directory", default="Twitter Dataset_CleanOutput", help="Name of the directory the tweet files are in")
  parser.add_argument("tweet_column_name", default="Tweet text_Clean", help="Name of the column the tweet text is in")
  parser.add_argument("sentiment_label_column_name", default="SentLabel", help="Name of column the sentiment label is in")
  args = parser.parse_args()

  dir_name = args.file_directory
  tweet_col = args.tweet_column_name
  sent_label_col = args.sentiment_label_col_name

  main(dir_name, tweet_col, sent_label_col)