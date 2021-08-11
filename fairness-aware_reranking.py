# Baseline Model
# Implements Absolute rank and bucket rank base on
# Only rank and bucket based on relevance score

from relevance import compute_relevance_cost
from scipy import special
import sys
import pandas as pd
import json
import numpy.random as nprand

seed = 1234

def compute_gender_distribution(list_of_genders):
    nprand.seed(seed)
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
            draw = nprand.choice(["Male","Female"],1,p=[0.5,0.5])
            if draw[0] == "Male":
                male+=1
            else:
                female+=1
            #male += 1
            #female += 1
    male_p = male / float(male + female)
    female_p = female / float(male + female)
    return [male_p, female_p]


def compute_cost_of_adding(relevance, ranked_list_dist, corpus_dist, w_r, w_g):
    if ranked_list_dist == []:
        kl_div = 0
    else:
        kl_div = sum(special.kl_div(ranked_list_dist, corpus_dist))
    cost = (w_r * relevance) + (w_g * kl_div)

    return cost


def get_relevance_and_fairness_based_ranking(df, w_r, w_g):

    detected_objects_map = pd.Series(
        df["Extracted_Labels"].values, index=df["Image_Names"]
    ).to_dict()
    gender_map = pd.Series(df["Gender"].values, index=df["Image_Names"]).to_dict()
    terms = set(df["Term"].tolist())
    corpus = list(detected_objects_map.keys())
    corpus_gender_dist = compute_gender_distribution(df["Gender"].tolist())

    # Compute all relevance score once and store them for efficiency

    relevance_map = {}
    for image in corpus:
        objects = detected_objects_map[image]
        objects = objects.strip()
        objects = objects.split(",")
        relevance_cost = compute_relevance_cost(objects, terms)
        relevance_map[image] = relevance_cost

    ranked_list_with_scores = sorted(relevance_map.items(), key=lambda x: x[1])
    # implement the algorithm

    final_ranked_list = []
    gender_values_so_far = []

    while len(corpus) != 0:
        c_min = 10000
        d_min = None
        gender_to_be_added = None
        for d in corpus:
            relevance_score = relevance_map[d]
            gender_of_current_document = gender_map[d]

            ranked_list_dist = compute_gender_distribution(
                gender_values_so_far + [gender_of_current_document]
            )
            c = compute_cost_of_adding(
                relevance_score, ranked_list_dist, corpus_gender_dist, w_r, w_g
            )
            if c < c_min:
                d_min = d
                c_min = c
                gender_to_be_added = gender_of_current_document

        # append the one with minimum cost and remove it from the corpus
        final_ranked_list.append(d_min)
        gender_values_so_far.append(gender_to_be_added)
        corpus.remove(d_min)   
    return final_ranked_list


def main():
    filename = sys.argv[1]
    output_dir = sys.argv[2]

    w_r_list = [0.1, 0.3, 0.5, 0.7,0.9]

    for w_r in w_r_list:
        df = pd.read_excel(filename, sheet_name="Sheet1")
        w_g = round(1 - w_r, 1)  # alleviate some weird floating point subtraction issue
        relevance_with_fairness_ranks = get_relevance_and_fairness_based_ranking(
            df, w_r, w_g
        )
        with open(output_dir + "/" + f"relevance_{w_r}_fairness_{w_g}.txt", "w") as f:
            f.write("\n".join(relevance_with_fairness_ranks))


if __name__ == "__main__":
    main()
