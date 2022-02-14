import os
import pandas as pd
from sklearn.metrics import accuracy_score
import argparse
import csv

# The main routine
def main(dir_name: str, cluster_pipe1: str, cluster_pipe3: str) -> None:
  accuracies = []

  # Bootstrap each language
  for rf in sorted(os.listdir(dir_name)):
    df = pd.read_csv(dir_name + "/" + rf)

    accuracies = bootstrap(df, cluster_pipe3, cluster_pipe1)

    # Save the accuracies to csv
    path = f"BootstrappedCluster/"
    os.makedirs(path, exist_ok=True)  # Create the directory if it doesn't exist
    with open(path + rf, "w") as wf:
      writer = csv.writer(wf)
      writer.writerow(accuracies)
  

def bootstrap(df, pred_label_col: str, true_label_col: str):
  accuracies = []

  for i in range(1000):
    if i % 10 == 0:
      print(i)

    sample_df = df.sample(frac=1, replace=True, random_state=i)  # Sample the same number of rows as original df, with replacement
    accuracies.append(accuracy_score(sample_df[true_label_col], sample_df[pred_label_col]))
  
  return accuracies


if __name__ == "__main__":
  # parser = argparse.ArgumentParser()
  # parser.add_argument("file_directory", default="EnglishToOriginalTweets_TopicClusterOutput", help="Name of the directory the tweet files are in")
  # parser.add_argument("cluster_column_name_pipe1", default="Tweet text_Clean_Cluster", help="Name of the column the cluster label is in for pipeline 1")
  # parser.add_argument("cluster_column_name_pipe3", default="ReverseTrans_Cluster", help="Name of the column the cluster label is in for pipeline 3")
  # args = parser.parse_args()

  # dir_name = args.file_directory
  # cluster_pipe1 = args.cluster_column_name_pipe1
  # cluster_pipe3 = args.cluster_column_name_pipe3

  dir_name = "EnglishToOriginalTweets_30000Sample_TopicClusterOutput"
  cluster_pipe1 = "Tweet text_Clean_Cluster"
  cluster_pipe3 = "ReverseTrans_Cluster"
  main(dir_name, cluster_pipe1, cluster_pipe3)
