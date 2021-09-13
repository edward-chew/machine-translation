import os
import re
import pandas as pd


def main(dir_name: str, tweet_col: str) -> None:
  print("--- Cleaning ---")
  # Go through all the files in the specified directory
  for rf in sorted(os.listdir(dir_name)):
    print(rf)

    df = pd.read_csv(dir_name + "/" + rf)
    
    language = rf.split("_")[0]

    df[f"{tweet_col}_Clean"] = df.apply(lambda row: clean_text(row.get(tweet_col)), axis = 1)

    # Output to file
    output_dir_name = dir_name + "_CleanOutput"
    if not os.path.exists(output_dir_name):
      os.makedirs(output_dir_name)
    df.to_csv(output_dir_name + "/" + language + "_clean_tweets.csv", index = False)


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
  dir_name = "Twitter Dataset"  # Name of the directory the tweet files are in
  tweet_col = "Tweet text"  # Name of the column the tweet text is in
  main(dir_name, tweet_col)