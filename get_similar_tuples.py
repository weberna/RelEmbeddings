###########################################
#    For some tuple, compute 
#    the top k most similar other relations
###########################################
import numpy as np
from scipy.spatial.distance import cosine
import pickle
import sys
from tuple_reader import TupleData


REL_COUNT = 15743
WORD_COUNT = 48008

def most_similar(embed_dict, tuple_data, target, k, rev_rel_dict, rev_word_dict):
    """
    Return the k most similar tuples to the target
    (target should be an index to tuple_data.tuples), embed_dict maps
    indexes into tuple_data.tuples to its full embedding
    """
    tuples = tuple_data.tuples
    dist_dict = dict.fromkeys(embed_dict.keys(), -1) #dictionary that maps relation to distance from target
    targ_vect = tuples[target].toarray()[0]

    print("TARGET: {}".format(hot3_to_tuple(targ_vect, rev_rel_dict, rev_word_dict)))


    for rel_index in dist_dict.keys():
        other_rel = embed_dict[rel_index]
        dist = cosine(embed_dict[target], other_rel)
        dist_dict[rel_index] = dist

    
    li = []
    for item in sorted(dist_dict.items(), key=lambda tup: tup[1]):
        tup = tuples[item[0]].toarray()[0]
        tup_txt = hot3_to_tuple(tup, rev_rel_dict, rev_word_dict)
        li.append((tup_txt, item[1]))


    return li[:k]


def get_tuple_embeddings(embeds, tuple_data, rel_dict, word_dict):
    """
    Return a dictinary that maps index into tuples (TupleData object with 3hot vect)
    to its corresponding full embedding gotton through matmul with embeds
    """
    tuples = tuple_data.tuples
    embed_dict = dict.fromkeys(range(tuples.shape[0]), 0)
    for i in range(tuples.shape[0]):
        hot_vec = tuples[i].toarray()
        embedding = np.matmul(hot_vec, embeds)
        embed_dict[i] = embedding
    return embed_dict
        

def hot3_to_tuple(hot3, rev_rel_dict, rev_word_dict):
    words = len(rev_word_dict.keys())
    rels = len(rev_rel_dict.keys())
#    hot3 = hot3.toarray()[0]

    a1 = hot3[:words]
    rel = hot3[words:words + rels]
    a2 = hot3[words+rels:]


    a1_index = np.where(a1==1)[0][0] #find the index that contains a 1
    rel_index = np.where(rel==1)[0][0] #find the index that contains a 1
    a2_index = np.where(a2==1)[0][0] #find the index that contains a 1


    a1_txt = rev_word_dict[a1_index]
    rel_txt = rev_rel_dict[rel_index]
    a2_txt = rev_word_dict[a2_index]

    return "({}, {}, {})".format(a1_txt, rel_txt, a2_txt)

if __name__ == "__main__":
    embedfile = sys.argv[1] #full embedding file
    tuple_file = sys.argv[2]
    relfile = sys.argv[3] #rel dict
    wordfile = sys.argv[4] #word dict
#    targ = sys.argv[3]  
    
    embeds = pickle.load(open(embedfile, 'rb'))
    rel_dict = pickle.load(open(relfile, 'rb'))
    word_dict = pickle.load(open(wordfile, 'rb'))
    tuple_data =  pickle.load(open(tuple_file, 'rb'))

    tuple_data.tuples = tuple_data.tuples[:10000]
    

    rev_rel_dict = dict(zip(rel_dict.values(),rel_dict.keys()))
    rev_word_dict = dict(zip(word_dict.values(),word_dict.keys()))
    embed_dict = get_tuple_embeddings(embeds, tuple_data, rel_dict, word_dict)


    print(most_similar(embed_dict, tuple_data, 0, 10, rev_rel_dict, rev_word_dict))

