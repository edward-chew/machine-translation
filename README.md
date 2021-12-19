# Polyglot Sentiment
## Background
Designed for the dataset from *Multilingual Twitter Sentiment Classification: The Role of Human Annotators (Mozetič, Igor, et al.).* Paper found [here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155036). Tweet ID dataset found [here](https://www.clarin.si/repository/xmlui/handle/11356/1054). 

The input files of the scripts are files with the full tweet text, not tweet ID's.

## Contents
This repository contains the following:
- **clean.py**
    - Cleans the tweet text: removing return handles, Twitter handles, and number as well as makeing all text lowercase.
- **sentiment.py**
    - Gets the [Polyglot](https://polyglot.readthedocs.io/en/latest/Sentiment.html) sentiment of tweets, and converts them to text labels (Negative, Neutral, Positive).
- **lang_codes.py**
    - The language codes used by sentiment.py to provide language hints to Polyglot.
- **translate.py**
    - *(To be added)* Translates tweet text between English and the 15 other specified lanaguges, and vice versa.
- **bootstrap.py**
    - Bootstraps the accuracies of the sentiment classification.
- **results.ipynb**
    - Generates figures.

## Usage
1. **Clean the tweets.**

        python clean.py [file_directory] [tweet_column_name]
    where `[file_directory]` is the directory of the tweet files and `[tweet_column_name]` is the name of the csv column of the tweets.

    The outputs a directory `/[file_directory]_CleanOutput`.
2. **Translate and back translate the tweets.**

        python translate.py [file_directory] [tweet_column_name]
3. **Get the sentiment.**

        python sentiment.py [file_directory] [tweet_column_name] [true_label_column_name]
    where `[file_directory]` is the directory of the tweet files (outputs of `clean.py` and `translate.py`) and `[true_label_column_name]` is the name of the csv column of the correct sentiment label.

    The outputs a directory `/[file_directory]_PolyglotSentimentOutput`.
4. **Bootstrap the results.**

        python bootstrap.py [file_directory] [poly_label_column_name] [true_label_column_name]
    where `[file_directory]` is the directory of the tweet files with sentiment labels, `[poly_label_column_name]` is the name of the csv column of the Polyglot sentiment label, and `[true_label_column_name]` is the name of the csv column of the correct sentiment label.

    This outputs a directory `/Bootstrapped` of structure

        Bootstrapped
        ├── EnglishToOriginalTweets
        │   ├── AllTweets
        │   │   ├── Albanian.csv
        │   │   ├── ... All other languages
        │   │   └── Swedish.csv
        │   └── NoNeuTweets
        │       ├── Albanian.csv
        │       ├── ...
        │       └── Swedish.csv
        ├── TranslatedToEnglishTweets
        └── Twitter\ Dataset
    `/TranslatedToEnglishTweets` and `/Twitter Dataset` have the same structure as `/EnglishToOriginalTweets`. `/NoNeuTweets` contains accuracies excluding true Neutral tweets and tweets labeled Neutral, while `AllTweets` contains accuracies using all tweet types.
5. **Generate figures.** Rerun all cells of `results.ipynb`. Image files are saved to directory `/Figures`.


## Languages
Languages involved are Albanian, Bulgarian, English, German, Hungarian, Polish, Portuguese, Russian, Ser/Cro/Bos (Serbian, Croatian, and Bosnian), Slovak, Slovenian, Spanish, and Swedish.  
Language codes are sq, bg, en, de, hu, pl, pt, ru, sh (sr, hr, bs), sk, sl, es, sv, respectively.