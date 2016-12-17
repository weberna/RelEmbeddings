import numpy as np
import pickle
from tuple_reader import TupleData
import sys

def print_rels(tup_data, rel_dict, word_dict, outfile):
    """
    print out the tuples in tup_data into outfile
    """
    words = len(word_dict.keys())
    rels = len(rel_dict.keys())
    reverse_rels = dict(zip(rel_dict.values(), rel_dict.keys()))
    out = open(outfile, 'w')
    for tup in tup_data.tuples:
        tup = tup.toarray()[0] #get numpy version of tuple
        rel_array = tup[words:words+rels]
        index = np.where(rel_array==1)[0][0] #find the index that contains a 1
        rel = reverse_rels[index]
        rel = rel.replace(' ', '_')
        out.write("{} ".format(rel))

if __name__ == "__main__":
    tup_data = pickle.load(open("TupleDataOut.pkl", 'rb'))
    rel_dict = pickle.load(open("rel.pkl", 'rb'))
    word_dict = pickle.load(open("word.pkl", 'rb'))
    outfile = sys.argv[1]
    print_rels(tup_data, rel_dict, word_dict, outfile)
