import pickle as pkl
import numpy as np
import sys

def init_weights(word_dict, other_dict, orig_embed, embed_size, using_rels=False):
    weights = np.random.uniform(-1, 1, (len(word_dict.keys()), embed_size))
    for word in word_dict.keys():  #set the weights for all words in dictionary
        if using_rels: #if we are using relations, must put into corret format
            word = word.replace(' ', '_')
        if word in other_dict.keys(): #if this word has an embedding in orig_embed
            print("YEAH")
            weights[word_dict[word]] = orig_embed[other_dict[word]] #init this embedding 
    return weights


if __name__ == "__main__":
    word_file = sys.argv[1]
    other_dict_file = sys.argv[2]
    embed_file = sys.argv[3]
    out_file = sys.argv[4]
    use_rels = sys.argv[5]
    word_dict = pkl.load(open(word_file, "rb"))
    embed_dict = pkl.load(open(other_dict_file, "rb"))
    embeddings = pkl.load(open(embed_file, "rb"))
    words_size, embed_size = embeddings.shape
    if use_rels == "yes":
        weights = init_weights(word_dict, embed_dict, embeddings, embed_size, True)
    else:
        weights = init_weights(word_dict, embed_dict, embeddings, embed_size)
    pkl.dump(weights, open(out_file,"wb"))
