import os
import pandas as pd
import numpy as np
import googletrans
from lang_codes import lang_codes

def main(dir_name):
  # Go through all the files in the specified directory
  for rf in sorted(os.listdir(dir_name)):
    print(rf)

    df = pd.read_csv(dir_name + "/" + rf)
    
    language = rf.split("_")[0]

    df["TranslatedToEnglish"] = df.apply(lambda row: translate(row.get("CleanTweetText"), lang_codes[language], "en", row.name), axis = 1)

    df.to_csv("TranslatedToEnglishTweets/" + language + "_to_English_tweets.csv", index = False)


def translate(text: str, src: str, dest: str, index) -> str:
  if index % 1000 == 0:
    print(index)

  if type(text) != str:
    return ""
  
  translator = googletrans.Translator()
  translated = translator.translate(text, src=src, dest=dest)
  return translated.text


if __name__ == "__main__":
  dir_name = "CleanTweetsTester"
  main(dir_name)