from polyglot.text import Text
from polyglot.detect.base import logger as polyglot_logger
import csv
import os
import pandas as pd
import numpy as np

lang_codes = {
  "Albanian" : "sq",
  "Bulgarian": "bg",
  "English": "en",
  "German": "de",
  "Hungarian": "hu",
  "Polish": "pl",
  "Portuguese": "pt",
  "Russian": "ru",
  "Serbo-Croatian": "sh",
  "Serbian": "sr",
  "Croatian": "hr",
  "Bosnian": "bs",
  "Slovak": "sk",
  "Slovenian": "sl",
  "Spanish": "es",
  "Swedish": "sv"
}

# The main routine
def main(dir_name):
  # Silence error messages
  polyglot_logger.setLevel("ERROR")

  # Go through all the files in the specified directory
  for rf in sorted(os.listdir(dir_name)):
    print(rf)

    df = pd.read_csv(dir_name + "/" + rf)

    language = rf.split("_")[0]

    df["PolyglotSentiment"] = df.apply(lambda row: get_polarity(row.CleanTweetText, language), axis = 1)
    df["PolyglotLabel"] = df.apply(lambda row: get_label(row.PolyglotSentiment), axis = 1)

    df.to_csv("PandasOutput/" + language + "_polyglot.csv", index = False)


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

  # Detected language must match actual language
  # if lang == "Serbian" or lang == "Bosnian" or lang == "Croatian":
  #   if t.language.code != lang_codes["Serbian"] \
  #     or t.language.code != lang_codes["Bosnian"] \
  #     or t.language.code != lang_codes["Croatian"] \
  #     or t.language.code != lang_codes["Serbo-Croatian"]:
  #     return np.nan
  # elif t.language.code != lang_codes[lang]:
  #   return np.nan

  assert(t.language.code == lang_codes[lang])

  # Sentiment detector fails when polarity is 0
  try:
    return t.polarity
  except ZeroDivisionError as e:
    return 0


if __name__ == "__main__":
  dir_name = "CleanTweets"
  main(dir_name)
