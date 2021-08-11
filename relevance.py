# Compute w2v based similarity
# Install gensim for this
import sys
import gensim.downloader as api
import numpy as np
from scipy import spatial
from gensim.models import KeyedVectors

# Initialize w2v model
try:
    w2v_model = KeyedVectors.load_word2vec_format(
        "glove-wiki-gigaword-300-binary", binary=True
    )
except:
    print("Binary model not present. Loading and saving general model")
    w2v_model = api.load("glove-wiki-gigaword-300")
    w2v_model.save_word2vec_format("glove-wiki-gigaword-300-binary", binary=True)


def compute_relevance_cost(object_list, term_list):
    """
    object_list: a list of objects detected from the image (e.g., ["beaker", "person", "microscope"])
    term_list: a list of seaerch terms (e.g., ["biologist", "a person who studies biology"]
    """
    object_words = []
    term_words = []
    for ol in object_list:
        ol = ol.strip().replace("_", " ").lower()
        for word in ol.split():
            if word in w2v_model.key_to_index:
                object_words.append(word) 

    for tl in term_list:
        tl = tl.strip().replace("_", " ").lower()
        for word in tl.split():
            if word in w2v_model.key_to_index:
                term_words.append(word)

    #print("Printing object_words")
    #print(object_words)
    #print("Printing term words")
    #print(term_words)
    relevance_cost = 1 - w2v_model.n_similarity(object_words, term_words)
    return relevance_cost


def main():
    # Unit test
    # objects = ['Person','Human','Advertisement','Poster','Flyer','Brochure','Paper','Reading','Kindergarten']
    # terms = ["Primary School Teacher"]#, "a person who studies biology"]
    # print (compute_relevance_cost(objects, terms))

    terms = ["Biologist"]
    filename = sys.argv[1]
    file1 = open(filename, "r")

    d1 = {}
    for line in file1:
        line = line.strip()
        list1 = line.split(",", 1)
        d1[list1[0]] = list1[1].lstrip("[").rstrip("]").split(",")

    d2 = dict.fromkeys(d1.keys(), [])

    for i in d1.keys():
        objects = d1[i]
        # print("printing objects in d1 dict")
        relevance_cost = compute_relevance_cost(objects, terms)
        d2[i] = relevance_cost

    # now sort from high to low

    sorted_list = sorted(d2.items(), key=lambda x: x[1])

    for item in sorted_list:
        print(item)
    file1.close()


if __name__ == "__main__":
    main()
