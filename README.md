# Machine Translation Demonstrating the Preservation of Semantic Content Across Multi-Language Dataset for use in State of the Art English-Trained Tools

## Datasets
This project involves 18 languages.

The datasets for Albanian, Bulgarian, English, German, Hungarian, Polish, Portuguese, Russian, Ser/Cro/Bos (Serbian, Croatian, and Bosnian), Slovak, Slovenian, Spanish, and Swedish are from *Multilingual Twitter Sentiment Classification: The Role of Human Annotators (Mozetič, Igor, et al.).* Paper found [here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155036). Tweet ID dataset found [here](https://www.clarin.si/repository/xmlui/handle/11356/1054). 

The datasets for Arabic, Chinese, and Hindi are from *TBCOV: Two Billion Multilingual COVID-19 Tweets with Sentiment, Entity, Geo, and Gender Labels (Imran, Muhammad, et al.)*. Paper found [here](https://www.mdpi.com/2306-5729/7/1/8). Tweet ID dataset found [here](https://crisisnlp.qcri.org/tbcov).

The input files of the scripts are files with the full tweet text, not tweet ID's.

## Contents
This repository contains the following:
- **clean.py**
    - Cleans the tweet text: removing return handles, Twitter handles, and number as well as makeing all text lowercase.
- **translate.py**
    - Translates tweet text from the 17 specified languages to English.
- **back_translate.py**
    - Translates tweet text from English back to the 17 specified languages.
- **results.ipynb**
    - Generates figures.
- **sample.py**
    - Generates randomly sampled subsets of csv files.

### Sentiment Task
- **sentiment.py**
    - Gets the [Polyglot](https://polyglot.readthedocs.io/en/latest/Sentiment.html) sentiment of tweets, and converts them to text labels (Negative, Neutral, Positive).
- **lang_codes.py**
    - The language codes used by sentiment.py to provide language hints to Polyglot.
- **bootstrap_sentiment.py**
    - Bootstraps the accuracies of the sentiment classification.

### Topic Clustering Task
- **cluster.py**
    - Assigns clusters to each tweet with [GSDMM](https://github.com/rwalk/gsdmm), once for Pipeline 1 and once for Pipeline 3.

### Word Embedding Task
- **word_embedding.py**
    - Calculates the Euclidean distance between the Pipeline 1 and Pipeline 3 tweets.

## Usage
1. **Sample the dataset.**

        python sample.py [file_directory] [sample_size]
    where `[file_directory]` is the directory of the tweet files and `[sample_size]` is number of tweets to sample.

    The outputs a directory `/[file_directory]_[sample_size]Sample`.
2. **Clean the tweets.**

        python clean.py [file_directory] [tweet_column_name] [sentiment_label_column_name]
    where `[file_directory]` is the directory of the tweet files, `[tweet_column_name]` is the name of the csv column of the tweets, and `[sentiment_label_column_name]` is the name of the csv column of the correct sentiment label.

    The outputs a directory `/[file_directory]_CleanOutput`.
3. **Translate and back translate the tweets.**

        python translate.py [file_directory] [tweet_column_name]
        python back_translate.py [file_directory] [tweet_column_name]
    where `[file_directory]` is the directory of the tweet files (outputs of `clean.py` and `translate.py`) and `[tweet_column_name]` is the name of the csv column of the tweets.
4. **Run semantic anaysis.**

    <ins>Sentiment Analysis</ins>

        python sentiment.py [file_directory] [tweet_column_name] [true_label_column_name]
    where `[file_directory]` is the directory of the tweet files (outputs of `clean.py` and `translate.py`) and `[true_label_column_name]` is the name of the csv column of the correct sentiment label.

    The outputs a directory `/[file_directory]_PolyglotSentimentOutput`.

    <ins>Topic Clustering</ins>

        python cluster.py [file_directory] [tweet_column_name_pipe1] [tweet_column_name_pipe3]
    where `[file_directory]` is the directory of the tweet files (files with both the Pipeline 1 and Pipeline 3 text) and `[tweet_column_name_pipe1]` and `[tweet_column_name_pipe1]` are the names of the csv columns of the Pipeline 1 and Pipeline 3 tweet texts, respectively.

    The outputs two directories:
    1. `/[file_directory]_kTopicClusterOutput` with the main results
    2. `/[file_directory]_ClusterBestModels` with the saved models

    <ins>Word Embedding</ins>

        python word_embedding.py [file_directory] [tweet_column_name_pipe1] [tweet_column_name_pipe3]
    where `[file_directory]` is the directory of the tweet files (files with both the Pipeline 1 and Pipeline 3 text) and `[tweet_column_name_pipe1]` and `[tweet_column_name_pipe1]` are the names of the csv columns of the Pipeline 1 and Pipeline 3 tweet texts, respectively.

    The outputs a directory `/[file_directory]_EmbeddingsOutput`.

5. **Bootstrap the results.**

    <ins>Sentiment Analysis</ins>

        python bootstrap_sentiment.py [file_directory] [poly_label_column_name] [true_label_column_name] [skip_english]
    where `[file_directory]` is the directory of the tweet files with sentiment labels, `[poly_label_column_name]` is the name of the csv column of the Polyglot sentiment label, `[true_label_column_name]` is the name of the csv column of the correct sentiment label, and `[skip_english]` indicates whether to skip the `English.csv` file.

    Run the script once for each pipeline, which adds a subdirectory to `/BootstrappedSentiment`.

    This outputs a directory `/[file_directory]_BootstrappedResults` of structure

        BootstrappedSentiment
        ├── ReverseTrans_Label
        │   ├── AllTweets
        │   │   ├── Albanian.csv
        │   │   ├── ... All other languages
        │   │   └── Swedish.csv
        │   └── NoNeuTweets
        │       ├── Albanian.csv
        │       ├── ...
        │       └── Swedish.csv
        ├── TranslatedToEnglish_Label
        └── Tweet text_Clean_Label
    `/TranslatedToEnglishTweets` and `/Twitter Dataset` have the same structure as `/EnglishToOriginalTweets`. `/NoNeuTweets` contains accuracies excluding true Neutral tweets and tweets labeled Neutral, while `AllTweets` contains accuracies using all tweet types.

6. **Generate figures.** Rerun all cells of `results.ipynb`. Image files are saved to directory `/Figures`.
