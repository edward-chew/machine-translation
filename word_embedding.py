import os
import argparse
from types import NoneType
from polyglot.mapping import Embedding
from lang_codes import lang_codes
import numpy as np
import pandas as pd
from scipy.spatial import distance
from polyglot.text import Text
import time

def sentence_embedding(embeddings, sentence : str, language : str) -> np.array:
    """Returns the sentence embedding (sum of word embeddings)

    Args:
        embeddings (Embedding): the Embeddings object
        sentence (str): sentence
        language (str): name of the language sentece is in
    """
    sent = Text(sentence, hint_language_code=lang_codes[language])
    sent_embedding = np.zeros(64)
    for word in sent.words:
        word_embedding = embeddings.get(word)
        if word_embedding is not None:
            sent_embedding = np.add(sent_embedding, word_embedding)
    return sent_embedding


def sentence_cosine_distance(embeddings, language : str, sentence1 : str, sentence2 : str) -> list[float]:
    """Returns the cosine distance between two sentences

    Args:
        embeddings (Embedding): the Embeddings object
        langauge (str): full name of the language of sentences
        sentence1 (str): sentence
        sentence2 (str): sentence

    Returns:
        float: Cosine distance or np.nan if either sentence is empty
    """
    if isinstance(sentence1, (float, NoneType)) or isinstance(sentence2, (float, NoneType)):
        return None

    sent1 = sentence_embedding(embeddings, sentence1, language)
    sent2 = sentence_embedding(embeddings, sentence2, language)
    
    return distance.cosine(sent1, sent2)


def baseline_distance(embeddings, language : str, df : pd.DataFrame, sentence_col : str) -> tuple[float, float]:
    """Calculates baseline distance for embeddings for each language

    Args:
        embeddings (Embedding): the Embeddings object
        langauge (str): full name of the language of sentences
        df (pd.DataFrame): DataFrame with embedding distances
        sentence_col (str): Name of the column sentence text is in

    Returns:
        float: Average minimum distance
        float: Average mean distance
    """
    subset = df.sample(5000, random_state=1)

    stats = subset.apply(lambda row: min_avg_distance(embeddings, language, subset, row.get(sentence_col), sentence_col), axis=1)

    mins = stats.apply(lambda x: x['Min'])
    means = stats.apply(lambda x: x['Mean'])

    return mins.mean(), means.mean()


def min_avg_distance(embeddings, language : str, df : pd.DataFrame, sentence : str, sentence_col : str) -> dict:
    """Returns the minimum and average distance between the sentence and all other sentences in the DataFrame

    Args:
        df (pd.DataFrame): DataFrame
        sentence (str): Sentence to compare
        sentence_col (str): Name of the column sentence text is in
    
    Returns:
        dict: Includes 'Min' and 'Mean'
    """

    dists = df.apply(lambda row: sentence_cosine_distance(embeddings, language, sentence, row.get(sentence_col)), axis=1)
    dists = dists[dists != 0]

    return {
        'Min': dists.min(),
        'Mean': dists.mean()
    }


def main(dir_name : str, tweet_col_pipe1 : str, tweet_col_pipe3 : str) -> None:
    baseline_df = pd.DataFrame(columns=['Language', 'Baseline Min', 'Baseline Mean'])

    # Create directory if doesn't already exist
    output_dir_name = f'{dir_name}_EmbeddingsOutput'
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)

    # Go through all the files in the specified directory
    for rf in sorted(os.listdir(dir_name)):
        print(rf)

        df = pd.read_csv(dir_name + '/' + rf)
        language = rf.split('.')[0]

        if language == 'English':
            print('   skipping...')
            continue

        embeddings = Embedding.load(f'/Users/edwardchew/polyglot_data/embeddings2/{lang_codes[language]}/embeddings_pkl.tar.bz2')
        embeddings = embeddings.normalize_words()

        df['SentenceDistance'] = df.apply(lambda row: sentence_cosine_distance(embeddings, language, row.get(tweet_col_pipe1), row.get(tweet_col_pipe3)), axis = 1)

        # Drop rows without an embedding
        df = df.dropna(subset=['SentenceDistance'])

        # Output to file
        df.to_csv(output_dir_name + '/' + rf, index = False)

        # Print baseline distances
        stats = baseline_distance(embeddings, language, df, tweet_col_pipe1)
        baseline_df.loc[len(baseline_df.index)] = [language, stats[0], stats[1]]
    
    baseline_df.to_csv(output_dir_name + '/Baseline.csv', index = False)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("file_directory", default="Twitter Dataset_CleanOutput", help="Name of the directory the tweet files are in")
  parser.add_argument("tweet_column_name_pipe1", default="Tweet text_Clean", help="Name of the column the Pipeline 1 cleaned tweet text is in")
  parser.add_argument("tweet_column_name_pipe3", default="Tweet text_Clean", help="Name of the column the Pipeline 3 tweet text is in")
  args = parser.parse_args()

  dir_name = args.file_directory
  tweet_col_pipe1 = args.tweet_column_name_pipe1
  tweet_col_pipe3 = args.tweet_column_name_pipe3

  start_time = time.time()
  main(dir_name, tweet_col_pipe1, tweet_col_pipe3)
  print("--- %s seconds ---" % (time.time() - start_time))
