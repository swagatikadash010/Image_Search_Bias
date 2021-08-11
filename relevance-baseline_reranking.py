# Baseline Model
# Implements Absolute rank and bucket rank base on
# Only rank and bucket based on relevance score

import sys
import pandas as pd
import json
from relevance import compute_relevance_cost
import random

def get_relevance_based_ranking(df):
    detected_objects_map = pd.Series(
        df["Extracted_Labels"].values, index=df["Image_Names"]
    ).to_dict()
    terms = list(set(df["Term"].tolist()))
    # print ("TERMS",terms)
    relevance_map = {}

    for image in detected_objects_map.keys():
        objects = detected_objects_map[image]
        objects = objects.strip()
        objects = objects.split(",")
        relevance_cost = compute_relevance_cost(objects, terms)
        relevance_map[image] = relevance_cost

    # now sort from low to high in terms of scores
    ranked_list_with_scores = sorted(relevance_map.items(), key=lambda x: x[1])
    # print (ranked_list_with_scores)
    ranked_list = [entry[0] for entry in ranked_list_with_scores]
    return ranked_list


def main():
    filename = sys.argv[1]
    output_dir = sys.argv[2]
    df = pd.read_excel(filename, sheet_name="Sheet1")
    relevance_ranks = get_relevance_based_ranking(df)
    with open(output_dir + "/" + "baseline2_relevance_only.txt", "w") as f:
        f.write("\n".join(relevance_ranks))


if __name__ == "__main__":
    main()
