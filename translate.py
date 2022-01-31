import os
import pandas as pd
import numpy as np
import googletrans
from lang_codes import lang_codes

def main(input_dir_name, output_dir_name):
  # Go through all the files in the specified directory
  for rf in sorted(os.listdir(input_dir_name)):
    print(rf)

    df = pd.read_csv(input_dir_name + rf)
    
    language = rf.split("_")[0]

    df["TranslatedToEnglish"] = df.apply(lambda row: translate(row.get("CleanTweetText"), lang_codes[language], "en", row.name), axis = 1)

    df.to_csv(output_dir_name + language + "_to_English_tweets.csv", index = False)


def translate(text: str, src: str, dest: str, index) -> str:
  if index % 1000 == 0:
    print(index)

  if type(text) != str:
    return ""
  
  translator = googletrans.Translator()
  translated = translator.translate(text, src=src, dest=dest)

  return translated.text


if __name__ == "__main__":
  input_dir_name = "CleanTweetsTester/"
  output_dir_name = "TranslatedToEnglishTweets/"
  main(input_dir_name, output_dir_name)
