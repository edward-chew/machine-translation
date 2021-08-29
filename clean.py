import csv
import os
import re


def main(dir_name):
  # Go through all the files in the specified directory
  for f in sorted(os.listdir(dir_name)):
    with open(dir_name + "/" + f, "r") as rf:
      csv_reader = csv.reader(rf, delimiter=',')
      next(csv_reader)  # Skip the headers
      language = f.split("_")[0]

      with open("CleanTweets/" + language + "_clean_tweets.csv", "w") as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(["\"Tweet text\"", "SentLabel", "CleanTweetText"])
        for i, row in enumerate(csv_reader):
          if i % 25000 == 0:
            print(f"{language}: {i}")
            
          text = row[0]
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
            
          csv_writer.writerow(row[0:2] + [text])


if __name__ == "__main__":
  dir_name = "Twitter Dataset"
  main(dir_name)