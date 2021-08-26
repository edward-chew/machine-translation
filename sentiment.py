from polyglot.text import Text
from polyglot.detect.base import logger as polyglot_logger
import csv
import os

def main(dir_name):
  # Silence error messages
  polyglot_logger.setLevel("ERROR")

  failures = {}
  # Go through all the files in the specified directory
  for j, f in enumerate(sorted(os.listdir(dir_name))):
    with open(dir_name + "/" + f, "r") as rf:
      csv_reader = csv.reader(rf, delimiter=',')
      next(csv_reader)  # Skip the headers
      language = f.split("_")[0]

      failed_tweets = 0
      with open("Output/" + language + "_polyglot.csv", "w") as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(["\"Tweet text\"", "SentLabel", "PolyglotSentiment"])
        for i, row in enumerate(csv_reader):
          if i % 25000 == 0:
            print(f"{language}: {i}")
          try:
            text = Text(row[0].strip("\""))

            # Sentiment detector fails when polarity is 0
            polarity = 0
            try:
              polarity = text.polarity
            except ZeroDivisionError as e:
              polarity = 0

            sentiment = "Neutral"
            if polarity < 0:
              sentiment = "Negative"
            elif polarity > 0:
              sentiment = "Positive"
            
            csv_writer.writerow(row[:2] + [polarity, sentiment])
          except:
            failed_tweets += 1
            csv_writer.writerow(row[:2]  + ["", ""])
      failures[language] = failed_tweets

  print(failures)


if __name__ == "__main__":
  dir_name = "Twitter Dataset"
  main(dir_name)
