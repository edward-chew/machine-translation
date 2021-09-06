import csv
import os
import re
import pandas as pd


def main(dir_name):
  # Go through all the files in the specified directory
  for rf in sorted(os.listdir(dir_name)):
    print(rf)

    df = pd.read_csv(dir_name + "/" + rf)
    
    language = rf.split("_")[0]

    df["CleanTweetText"] = df.apply(lambda row: clean_text(row.get("Tweet text")), axis = 1)

    df.to_csv("CleanTweets/" + language + "_clean_tweets.csv", index = False)


def clean_text(text: str) -> str:
  # remove twitter Return handles (RT @xxx:)
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


if __name__ == "__main__":
  dir_name = "Twitter Dataset"
  main(dir_name)