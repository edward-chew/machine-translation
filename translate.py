import os
import six
import pandas as pd
from google.cloud import translate_v2 as translate
import argparse

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "apikey.json"


def translate_text(target, text):
    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    try:
        result = translate_client.translate(text, target_language=target)
        # print(u"Text: {}".format(result["input"]))
        # print(u"Translation: {}".format(result["translatedText"]))
        # print(u"Detected source language: {}".format(
        #     result["detectedSourceLanguage"]))
        return result["translatedText"]
    except:
        return ""


def main(dir_name, tweet_col):
    # Go through all the files in the specified directory
    for rf in sorted(os.listdir(dir_name)):
        print(rf)

        df = pd.read_csv(dir_name + "/" + rf)

        df["TranslatedToEnglish"] = df.apply(lambda row:
                                             translate_text("en", row.get(tweet_col)), axis=1)

        output_dir_name = dir_name + "_TranslatedToEnglish"
        if not os.path.exists(output_dir_name):
            os.makedirs(output_dir_name)
        df.to_csv(output_dir_name + "/" + rf, index=False)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("file_directory", default="Twitter Dataset_CleanOutput", help="Name of the directory the tweet files are in")
    # parser.add_argument("tweet_column_name", default="Tweet text_Clean", help="Name of the column the tweet text is in")
    # args = parser.parse_args()

    # dir_name = args.file_directory
    # tweet_col = args,tweet_column_name
    dir_name = "Twitter Dataset New Languages_CleanOutput_10000Sample"
    tweet_col = "Tweet text_Clean"

    main(dir_name, tweet_col)
