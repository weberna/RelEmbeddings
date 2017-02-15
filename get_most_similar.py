###########################################
#    For some relation, compute
#    the top k most similar other relations
###########################################
import numpy as np
from scipy.spatial.distance import cosine
import pickle
import sys

def most_similar(embed, target, k, rel_dict):
    """
    Return a list of indexes of the k most
    similar vectors for the relationship at index 
    target. rel_dict should map rels to indices
    """
    dist_dict = dict.fromkeys(rel_dict.keys(), -1) #dictionary that maps relation to distance from target
    targ_vect = embed[target][:]

    reverse_dict = dict(zip(rel_dict.values(),rel_dict.keys()))
    print("TARGET: {}".format(reverse_dict[target]))
    for rel in dist_dict.keys():
        other_rel = embed[rel_dict[rel]][:]
        dist = cosine(targ_vect, other_rel)
        dist_dict[rel] = dist

    
    li = []
    for item in sorted(dist_dict.items(), key=lambda tup: tup[1]):
        
        li.append(item)


    return li[:k]

if __name__ == "__main__":
    embedfile = sys.argv[1]
    relfile = sys.argv[2]
    targ = sys.argv[3]  
    
    embed = pickle.load(open(embedfile, 'rb'))
    rel_dict = pickle.load(open(relfile, 'rb'))
    target = rel_dict[targ]
    
    samples = np.random.random_integers(0, 15000, 40)
#    samples = np.arange(0,50)
    for i in samples:
        print(most_similar(embed, i, 10, rel_dict))
        print("---------------------------------")

    sim = most_similar(embed, target, 10, rel_dict)
    print(sim)
