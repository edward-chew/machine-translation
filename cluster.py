import pandas as pd
import numpy as np
import string
import re
import os
import gsdmm
import pickle
import stopwordsiso
import unicodedata
import argparse
import nltk
import time
from gsdmm import MovieGroupProcess
from tqdm import tqdm
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize
from nltk.parse import CoreNLPParser
from stopwordsiso import stopwords as iso_stopwords
from sklearn.metrics import accuracy_score
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from lang_codes import lang_codes

PUNKTLANGS = ["czech", "danish", "dutch", "english", "estonian", "finnish", "french", "german", "greek", "italian", "malayalam", "norwegian", "polish", "portuguese", "russian", "slovenian", "spanish", "swedish", "turkish", "arabic", "chinese"]


def remove_emojis(text):
  regrex_pattern = re.compile(pattern = "["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        "]+", flags = re.UNICODE)
  return regrex_pattern.sub(r"", text)


def tokenize(text, lang):
  """
  Returns text as a list of tokens, stopwords removed.
  
  text: text to tokenize
  lang: language of text, lowercase
  """

  # Checks for null strings
  if isinstance(text, float):
    return []

  text = remove_emojis(text)

  if lang.lower() in nltk_stopwords.fileids():
    stoplist = nltk_stopwords.words(lang.lower())
  else:
    stoplist = list(iso_stopwords(lang_codes[lang.title()]))
  
  # Chinese and Arabic use different tokenizer
  if lang == "chinese":
    parser = CoreNLPParser('http://localhost:9001')
    tokens = list(parser.tokenize(text))
  elif lang == "arabic":
    parser = CoreNLPParser('http://localhost:9005')
    tokens = list(parser.tokenize(text))
  elif lang == "slovenian":
    tokens = word_tokenize(text, language="slovene")
  else:
    tokens = word_tokenize(text, language=lang)

  # Only include tokens that aren't stop words, are more than 2 characters long, and are not punctuation marks
  tokens = [x for x in tokens if x not in stoplist and not unicodedata.category(x[0]).startswith("P")]
  if lang != "chinese":
    tokens = [x for x in tokens if len(x) > 2 and len(x) <= 15]
  
  return tokens


def topic_allocation(df: pd.DataFrame, docs, mgp, tweet_col):
  """allocates all topics to each document in original dataframe"""
  topic_allocations = []
  for doc in tqdm(docs):
    topic_label, _ = mgp.choose_best_label(doc)
    topic_allocations.append(topic_label)

  df.loc[:, tweet_col + "_Cluster"] = topic_allocations

  print("Complete. Number of documents with topic allocated: {}".format(len(df)))


# Generate labels from the model and write to the df
def get_labels(df: pd.DataFrame, tweet_col: str, model, language: str):
  df[tweet_col + "_Tokens"] = df[tweet_col].apply(tokenize, lang=language.lower())
  docs = df[tweet_col + "_Tokens"].tolist()
  topic_allocation(df, docs, model, tweet_col)


# Define function to get words in topics (to compute coherence score)
def get_topics_lists(model, top_clusters, n_words):
  """
  Gets lists of words in topics as a list of lists.
  
  model: gsdmm instance
  top_clusters:  numpy array containing indices of top_clusters
  n_words: top n number of words to include
  
  """
  # create empty list to contain topics
  topics = []
  
  # iterate over top n clusters
  for cluster in top_clusters:
    #create sorted dictionary of word distributions
    sorted_dict = sorted(model.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:n_words]
    
    #create empty list to contain words
    topic = []
    
    #iterate over top n words in topic
    for k,v in sorted_dict:
      #append words to topic list
      topic.append(k)
        
    #append topics to topics list    
    topics.append(topic)
  
  return topics


def get_best_model(df: pd.DataFrame, tweet_col: str, language: str, k_list: list[int], alpha_list: list[float], beta_list: list[float], dir_name: str):
  """Get the best model for a given language by conducting a grid search for hyperparameters K, alpha, and beta

  Args:
    df (pd.DataFrame): tweets
    tweet_col (str): name of column of tweets
    language (str): language of tweets
    k_list (list[int]): k values to test (number of potential clusters)
    alpha_list (list[float]): alpha values to test (controls completeness)
    beta_list (list[float]): beta values to test (controls homogeneity)
    dir_name (str): name of directory tweet files are in

  Returns:
    gsdmm model
  """
  docs = df[tweet_col].apply(tokenize, lang=language.lower()).tolist()

  best_model = best_k = best_alpha = best_beta = None
  best_coherence = 0

  for k in k_list:
    for a in alpha_list:
      for b in beta_list:
        print(f"--- Testing k={k} a={a} b={b} ---")
        mgp = gsdmm.MovieGroupProcess(K=k, alpha=a, beta=b, n_iters=5)
        dictionary = Dictionary(docs)
        n_terms = len(dictionary)
        mgp.fit(docs, n_terms)

        bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

        # Number of documents per topic
        doc_count = np.array(mgp.cluster_doc_count)

        print(doc_count)

        # Most important clusters (by number of docs inside), sorted from most to least important
        top_index = doc_count.argsort()[-10:][::-1]
        topics = get_topics_lists(mgp, top_index, 20) 

        # Get coherence and check if it is better than what we have so far
        cur_coherence = get_coherence(topics, dictionary, bow_corpus, docs)
        if cur_coherence > best_coherence:
          best_model, best_k, best_alpha, best_beta = mgp, k, a, b
          best_coherence = cur_coherence

  # Save the best model
  output_dir_name = dir_name + "_ClusterBestModels/"
  if not os.path.exists(output_dir_name):
    os.makedirs(output_dir_name)
  with open(output_dir_name + language + str(k) + ".model", "wb") as f:
    pickle.dump(best_model, f)
    f.close()
  
  print(f"\n--- Best model: k={best_k} alpha={best_alpha} beta={best_beta} ---\n")

  return best_model


# Get coherence score for a model
def get_coherence(topics, dictionary, bow_corpus, docs):
  for t in topics:
    if not t:
      raise ValueError("Topic has no tokens to define it")

  cm_gsdmm = CoherenceModel(topics=topics, 
                            dictionary=dictionary, 
                            corpus=bow_corpus, 
                            texts=docs, 
                            coherence='c_v')
  return cm_gsdmm.get_coherence()


def main(dir_name: str, tweet_col_pipe1: str, tweet_col_pipe3: str):
  # Go through all the files in the specified directory
  for rf in sorted(os.listdir(dir_name)):
    df = pd.read_csv(dir_name + "/" + rf)

    language = rf.split(".")[0]
    # Skip the langauge if the stopwords for preprocessing are not available
    if ((language.lower() not in nltk_stopwords.fileids() and lang_codes[language] not in stopwordsiso.langs())
        or language.lower() not in PUNKTLANGS
        or language.lower() == "english"):
      print(f"------ Skipping {language} ------")
      continue
    print(f"------ {language} ------")

    # Generate model from pipeline 1 clusters
    if True:
      for k in [2, 5, 10, 15, 20, 50, 100, 150, 200]:
        model = get_best_model(df, tweet_col_pipe1, language, [k], [0.1], [0.1], dir_name)

        # Generate cluster labels and append to df
        get_labels(df, tweet_col_pipe1, model, language)
        get_labels(df, tweet_col_pipe3, model, language)

        # Output to file
        output_dir_name = f"{dir_name}_kTopicClusterOutput/{k}"
        if not os.path.exists(output_dir_name):
          os.makedirs(output_dir_name)
        df.to_csv(output_dir_name + "/" + rf, index = False)

        # Load previously trained model instead
        if False:
          for k in [2, 10, 20, 50, 100]:
            df = pd.read_csv(dir_name + "/" + rf)
            f = open(f"{dir_name}_ClusterBestModels/{language}{k}.model", "rb")
            model = pickle.load(f)


if __name__ == "__main__":
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