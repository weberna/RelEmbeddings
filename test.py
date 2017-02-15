###########################################
#    For some tuple, compute 
#    the top k most similar other relations
###########################################
import numpy as np
from scipy.spatial.distance import cosine
import pickle
import sys
from tuple_reader import TupleData



def most_similar(embed_dict, tuple_data, target, k, rev_rel_dict, rev_word_dict):
    """
    Return the k most similar tuples to the target
    (target should be an index to tuple_data.tuples), embed_dict maps
    indexes into tuple_data.tuples to its full embedding
    """
    tuples = tuple_data.tuples
    dist_dict = dict.fromkeys(embed_dict.keys(), -1) #dictionary that maps relation to distance from target
    targ_vect = tuples[target].toarray()[0]
    targ_txt = hot3_to_tuple(targ_vect, rev_rel_dict, rev_word_dict)
    print("TARGET: {}".format(targ_txt))


    words = len(rev_word_dict.keys())
    rels = len(rev_rel_dict.keys())


    for rel_index in dist_dict.keys():
        other_rel = embed_dict[rel_index]
        dist = cosine(embed_dict[target], other_rel)
        dist_dict[rel_index] = dist

    
    li = [] #list of similar tuples
    li2 = []
    dists = sorted(dist_dict.items(), key=lambda tup: tup[1]) #list of tuples sorted by cosine distance to target
    #for item in sorted(dist_dict.items(), key=lambda tup: tup[1]):
    for item in dists:
        tup = tuples[item[0]].toarray()[0]
        tup_txt = hot3_to_tuple(tup, rev_rel_dict, rev_word_dict)
        if tup_txt not in li2 and tup_txt != targ_txt and not share_args(targ_vect,  tup, words, rels):
            li.append((tup_txt, item[1]))
            li2.append(tup_txt)
        if len(li) > k: 
            return li[:k]


    return li[:k]

def share_args(hot3_1, hot3_2, words, rels):

    a1 = hot3_1[:words]
    rel = hot3_1[words:words + rels]
    a2 = hot3_1[words+rels:]


    a1_index_1 = np.where(a1==1)[0][0] #find the index that contains a 1
    rel_index_1 = np.where(rel==1)[0][0] #find the index that contains a 1
    a2_index_1 = np.where(a2==1)[0][0] #find the index that contains a 1


    a1_2 = hot3_2[:words]
    rel_2 = hot3_2[words:words + rels]
    a2_2 = hot3_2[words+rels:]


    a1_index_2 = np.where(a1_2==1)[0][0] #find the index that contains a 1
    rel_index_2 = np.where(rel_2==1)[0][0] #find the index that contains a 1
    a2_index_2 = np.where(a2_2==1)[0][0] #find the index that contains a 1



    if a1_index_1 == a1_index_2 or rel_index_1 == rel_index_2  or a2_index_1 == a2_index_2 or a1_index_1 == a2_index_2 or a2_index_1 == a1_index_2:
        return True
    else:
        return False


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

    print(tuple_data.tuples.shape)
    tuple_data.tuples = tuple_data.tuples[:300000]
    

    rev_rel_dict = dict(zip(rel_dict.values(),rel_dict.keys()))
    rev_word_dict = dict(zip(word_dict.values(),word_dict.keys()))
    embed_dict = pickle.load(open("FULL_EMBED_DICT.pkl", 'rb'))

#    embed_dict = get_tuple_embeddings(embeds, tuple_data, rel_dict, word_dict)
    
    samples = np.random.random_integers(0, 299999, 50)
    for i in samples:
        print(most_similar(embed_dict, tuple_data, i, 20, rev_rel_dict, rev_word_dict))
        print("---------------------------------")
        print("\n")
  
#    print(most_similar(embed_dict, tuple_data, 4, 20, rev_rel_dict, rev_word_dict))


    
#    count = 0
#    for i in tuple_data.tuples:
#        hot3 = i.toarray()[0]
#        print("{} = {}".format(hot3_to_tuple(hot3, rev_rel_dict, rev_word_dict), count))
#        count += 1

#    print(most_similar(embed_dict, tuple_data, 3948, 10, rev_rel_dict, rev_word_dict))
#    print("---------------------------------")





