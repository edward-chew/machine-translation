import os
import pandas as pd
from lang_codes import lang_codes
from translate import translate_text
import argparse

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "apikey.json"


def main(dir_name, tweet_col):
    # Go through all the files in the specified directory
    for rf in sorted(os.listdir(dir_name)):
        print(rf)

        df = pd.read_csv(dir_name + "/" + rf)

        language = rf.split(".")[0]
        print(lang_codes[language])
        df["ReverseTrans"] = df.apply(lambda row:
                                             translate_text(lang_codes[language],
                                                            row.get(tweet_col)), axis=1)

        output_dir_name = "Twitter Dataset New Languages_CleanOutput_10000Sample_EnglishToOriginal"
        if not os.path.exists(output_dir_name):
            os.makedirs(output_dir_name)
        df.to_csv(output_dir_name + "/" + rf, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_directory", default="Twitter Dataset_CleanOutput", help="Name of the directory the tweet files are in")
    parser.add_argument("tweet_column_name", default="Tweet text_Clean", help="Name of the column the tweet text is in")
    args = parser.parse_args()

    dir_name = args.file_directory
    tweet_col = args.tweet_column_name

    main(dir_name, tweet_col)