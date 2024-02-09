from polyglot.text import Text
import pandas as pd
import os
import argparse
import regex
from polyglot.detect.base import logger as polyglot_logger
from typing import List

import sys
sys.path.append('../translation-project')
from lang_codes import lang_codes


def main(dir_name: str) -> None:
  polyglot_logger.setLevel("ERROR") # Suppress warning message about unreliable language detection

  for rf in sorted(os.listdir(dir_name)):
    lang = rf.split(".")[0]
    print(f"------ {lang} ------")

    df = pd.read_csv(dir_name + "/" + rf)

    print("Check for empty values")
    if lang == "English":
      contains_empty_values(df, ["SentLabel", "Tweet text_Clean"])
    else:
      contains_empty_values(df, ["SentLabel", "Tweet text_Clean", "TranslatedToEnglish", "ReverseTrans"])  

    print("Check detected language is expected language")
    if lang == "English":
      lang_matches_expected(df, ["Tweet text_Clean"], "English")
    else:
      lang_matches_expected(df, ["Tweet text_Clean", "ReverseTrans"], lang)
      lang_matches_expected(df, ["TranslatedToEnglish"], "English")


def contains_empty_values(df: pd.DataFrame, cols: List[str]) -> bool:
  """Check if any input columns have any empty values

  Args:
      df (pd.DataFrame): Dataframe
      cols (List[str]): Names of columns

  Returns:
      bool: Whether an empty value exists
  """
  for c in cols:
    if df[c].isnull().any():
      print(f"\tFAIL: Missing value in {c}")
      return True
  print("\tPASS")


def lang_matches_expected(df: pd.DataFrame, cols: List[str], lang: str) -> bool:
  """Check if more than 15% of cells within a column is in the wrong langauge, based on Polyglots lang detector

  Args:
      df (pd.DataFrame): Dataframe
      cols (List[str]): Names of columns
      lang (str): Expected language

  Returns:
      bool: Whether more than 15% wrong language
  """
  for c in cols:
    res = df[c].apply(correct_lang, expected_lang=lang)
    counts = res.value_counts()
    total = len(res)
    wrong_percent = round(counts[False] / total * 100, 2)
    if wrong_percent > 15:
      print("\tFAIL: >15% wrong language in {}.\t{}% or {} low confidence tweets out of {}".format(c, wrong_percent, counts[False], total))
    else:
      print(f"\tPASS {c}")


def correct_lang(s: str, expected_lang: str) -> bool:
  t = Text(remove_bad_chars(s))
  if (t.language.code.split("-")[0] != lang_codes[expected_lang] and 
      t.language.code.split("_")[0] != lang_codes[expected_lang]):
    return False
  return True


def remove_bad_chars(text):
  return regex.compile(r"[\p{Cc}\p{Cs}]+").sub("", text)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file_directory", help="Name of the directory the translated data is in")
  args = parser.parse_args()

  dir_name = args.file_directory

  main(dir_name)
