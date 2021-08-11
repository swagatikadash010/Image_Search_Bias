# Relevance is measured by bucket accuracy
# Fairness is measured by normalized KL divergence given in this paper
# https://arxiv.org/pdf/1905.01989.pdf

from scipy import special
from prepare_ground_truth import get_buckets_from_list
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os
import math
import numpy as np

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
    male_p = male / float(male + female)
    female_p = female / float(male + female)
    return [male_p, female_p]

def KL(list1, corpus_dist):
    dist = compute_gender_distribution(list1)
    kl_div = sum(special.kl_div(dist, corpus_dist))
    return kl_div
    
    
def normalized_discounted_kl_div(list1,corpus_dist):
    Z = sum([1/math.log(i+2,2) for i in range(len(list1))])
    NDKL = (1/Z) * sum([(1/math.log(i+2,2))*KL(list1[:i+1],corpus_dist) for i in range(len(list1))])
    return NDKL


def compute_bucket_ranking_accuracy(ranked_list,ground_truth_ranks):
    predicted_ranks = get_buckets_from_list(ranked_list)
    rank_pred = []
    rank_gold = []
    for document in predicted_ranks.keys():
        rank_pred.append(predicted_ranks[document])
        rank_gold.append(ground_truth_ranks[document])
    return accuracy_score(rank_pred, rank_gold)

def main():
    ground_truth_relevance_file = sys.argv[1]
    ground_truth_gender_file = sys.argv[2]
    runs_dir = sys.argv[3] 
    
    ground_truth_relevance = {}
    ground_truth_gender = {}
    with open(ground_truth_relevance_file) as f:
        for line in f:
            line = line.strip()
            document, rank = line.split(",")
            ground_truth_relevance[document.strip()] = int(rank)
    
    with open(ground_truth_gender_file) as f:
        for line in f:
            line = line.strip()
            document, gender = line.split(",")
            ground_truth_gender[document.strip()] = gender.strip()
    
    corpus_dist = compute_gender_distribution(list(ground_truth_gender.values()))
    runs_all = []
    
    relevance_data = {}
    fairness_data = {}
    for fn in os.listdir(runs_dir):
        if "txt" in fn:
            runs_all.append(fn.replace(".txt",""))
            ranked_list = []
            with open (runs_dir+"/"+fn) as f:
                for line in f:
                    line  = line.strip()
                    ranked_list.append(line)
            relevance = compute_bucket_ranking_accuracy(ranked_list, ground_truth_relevance)
            gender_list = [ground_truth_gender[document] for document in ranked_list]
            fairness = normalized_discounted_kl_div(gender_list, corpus_dist)            
            relevance_data[fn.replace(".txt","")] = relevance
            fairness_data[fn.replace(".txt","")] = fairness
    
    
    runs_all.sort()
    result_file = open("./results/results.txt","w")
    for rk in runs_all:
        result_file.write(rk+","+str(relevance_data[rk])+","+str(fairness_data[rk])+"\n")
    result_file.close()
    
    # now plot
    colormap = plt.cm.gist_ncar 
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(runs_all))]
    for i, run in enumerate(runs_all):
        plt.scatter(x=[fairness_data[run]], 
            y=[relevance_data[run]], c=[colors[i]],
            label=run, s=50)

    plt.legend(loc='center', fontsize='xx-small',bbox_to_anchor=(0.5, 1.05),ncol=len(runs_all)-3)
    txt = f"Population Dist P(male,female) = {corpus_dist}"
    plt.xlabel(f"Unfairness (NDKL)\n{txt}", fontsize='xx-small')
    plt.ylabel("Relevance")
    plt.savefig('./results/performance-plot.png')

if __name__=="__main__":
    main()
    exit(0)
    
