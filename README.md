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
    - *(To be added)* Translates tweet text between English and the 15 other specified lanaguges, and vice versa.
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
- **bootstrap_cluster.py**
    - Bootstraps the accuracies of the clustering.

### Word Embedding Task
- 

## Usage
1. **Sample the dataset.**

        python clean.py [file_directory] [sample_size]
    where `[file_directory]` is the directory of the tweet files and `[sample_size]` is number of tweets to sample.

    The outputs a directory `/[file_directory]_[sample_size]Sample`.
2. **Clean the tweets.**

        python clean.py [file_directory] [tweet_column_name] [sentiment_label_column_name]
    where `[file_directory]` is the directory of the tweet files, `[tweet_column_name]` is the name of the csv column of the tweets, and `[sentiment_label_column_name]` is the name of the csv column of the correct sentiment label.

    The outputs a directory `/[file_directory]_CleanOutput`.
3. **Translate and back translate the tweets.**

        python translate.py [file_directory] [tweet_column_name]
4. **Run semantic anaysis.**

    <ins>Sentiment Analysis</ins>

        python sentiment.py [file_directory] [tweet_column_name] [true_label_column_name]
    where `[file_directory]` is the directory of the tweet files (outputs of `clean.py` and `translate.py`) and `[true_label_column_name]` is the name of the csv column of the correct sentiment label.

    The outputs a directory `/[file_directory]_PolyglotSentimentOutput`.

    <ins>Topic Clustering</ins>

        python cluster.py [file_directory] [tweet_column_name_pipe1] [tweet_column_name_pipe3]
    where `[file_directory]` is the directory of the tweet files (files with both the Pipeline 1 and Pipeline 3 text) and `[tweet_column_name_pipe1]` and `[tweet_column_name_pipe1]` are the names of the csv columns of the Pipeline 1 and Pipeline 3 tweet texts, respectively.

    <ins>Word Embedding</ins>

5. **Bootstrap the results.**

    <ins>Sentiment Analysis</ins>

        python bootstrap_sentiment.py [file_directory] [poly_label_column_name] [true_label_column_name]
    where `[file_directory]` is the directory of the tweet files with sentiment labels, `[poly_label_column_name]` is the name of the csv column of the Polyglot sentiment label, and `[true_label_column_name]` is the name of the csv column of the correct sentiment label.

    This outputs a directory `/BootstrappedSentiment` of structure

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

    <ins>Topic Clustering</ins>

        python bootstrap_cluster.py [file_directory] [cluster_column_name_pipe1] [cluster_column_name_pipe3]
    where `[file_directory]` is the directory of the tweet files with cluster labels, and `[cluster_column_name_pipe1]` and `[cluster_column_name_pipe3]` are the names of the csv columns of the Pipeline 1 and Pipeline 3 tweets' cluster labels, respectively.

    This outputs a directory `/BootstrappedCluster`.

6. **Generate figures.** Rerun all cells of `results.ipynb`. Image files are saved to directory `/Figures`.
