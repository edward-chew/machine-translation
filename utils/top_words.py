import pandas as pd
import numpy as np
import argparse
import pickle
import os
from cluster import get_topics_lists

def main(dir_name):
  for rf in sorted(os.listdir(dir_name)):
    print(rf)

    rf_file = open(dir_name + "/" + rf, "rb")
    mgp = pickle.load(rf_file)

    # Number of documents per topic
    doc_count = np.array(mgp.cluster_doc_count)
    # Most important clusters (by number of docs inside), sorted from most to least important
    top_index = doc_count.argsort()[::-1]

    top_words = get_topics_lists(mgp, top_index, 20)

    df = pd.DataFrame()
    df["TopWords"] = top_words

    rf_base = rf.split(".")[0]
    output_dir_name = f"{dir_name}_TopWords"
    if not os.path.exists(output_dir_name):
      os.makedirs(output_dir_name)
    df.to_csv(output_dir_name + "/" + f"{rf_base}.csv", index = False)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file_directory", default="0_AllTranslationsCombined_Sampled_ClusterBestModels", help="Name of the directory the cluster models are in")
  args = parser.parse_args()

  dir_name = args.file_directory

  main(dir_name)