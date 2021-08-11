# Baseline Model
# Implements Absolute rank and bucket rank base on
# Only rank and bucket based on relevance score

import sys
import pandas as pd
import json
import random

seed = 1234
random.seed(seed)


def get_random_based_ranking(df):
    ranked_list = df["Image_Names"].tolist()
    for i in range(30):
        random.shuffle(ranked_list)
    return ranked_list


def main():
    filename = sys.argv[1]
    output_dir = sys.argv[2]
    df = pd.read_excel(filename, sheet_name="Sheet1")
    ranked_list = get_random_based_ranking(df)
    with open(output_dir + "/" + "baseline1_random.txt", "w") as f:
        f.write("\n".join(ranked_list))


if __name__ == "__main__":
    main()
