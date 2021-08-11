# We produce groundtruth and other files based on these guidelines from TREC 2019 task
# https://fair-trec.github.io/2019/doc/guidelines.pdf

import sys
import pandas as pd
import json
import numpy.random as nprand

seed = 1234
nprand.seed(seed)

RELEVANCE_BUCKET_SIZE = 10

def compute_gender_distribution(list_of_genders):
    if list_of_genders == []:
        return []
    male = 0
    female = 0
    for gender in list_of_genders:
        if gender.strip() == "Male":
            male += 1
        elif gender.strip() == "Female":
            female += 1
        else:
            male += 1
            female += 1
    male_p = male / float(male + female)
    female_p = female / float(male + female)
    return [male_p, female_p]

def get_buckets_from_list(list1):
    bucket_data = {}
    for i, document in enumerate(list1):
        bucket = int(i / RELEVANCE_BUCKET_SIZE)+1
        bucket_data[document] = bucket
    return bucket_data

def prepare_ground_truth_relevance(df):
    query = df["Term"][0].lower()
    gold_rank_map = pd.Series(df["GoldRank"].values, index=df["Image_Names"]).to_dict()
    document_data = []
    for image in gold_rank_map.keys():
        doc_id = image
        bucket = int(int(gold_rank_map[image]) / RELEVANCE_BUCKET_SIZE)+1
        document_data.append(doc_id+","+str(bucket))
    return document_data
    


def prepare_ground_truth_gender(df):
    df1 = df[["Image_Names", "GroundTruth_Gender"]]
    corpus_dist = compute_gender_distribution(df1['GroundTruth_Gender'].tolist())
    gender_data = []
    for i in df1.index:
        if df1["GroundTruth_Gender"][i].strip() == "Male":
            d = df1["Image_Names"][i] + ",Male"
        elif df1["GroundTruth_Gender"][i].strip() == "Female":
            d = df1["Image_Names"][i] + ",Female"
        else:
            draw = nprand.choice(["Male","Female"],1,corpus_dist)
            d = df1["Image_Names"][i] + ","+draw[0]
        gender_data.append(d)
    return gender_data


def main():
    filename = sys.argv[1]
    output_dir = sys.argv[2]
    df = pd.read_excel(filename, sheet_name="Sheet1")
    gt_relevance = prepare_ground_truth_relevance(df)
    with open(output_dir + "/" + "ground_truth_relevance.csv", "w") as f:
        f.write("\n".join(gt_relevance))
    gt_gender = prepare_ground_truth_gender(df)
    with open(output_dir + "/" + "ground_truth_gender.csv", "w") as f:
        f.write("\n".join(gt_gender))


if __name__ == "__main__":
    main()
