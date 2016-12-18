############################################################
#   tuple_reader.py
#   Functions that read in relational tuple data
#   convert them to 3 hot vectors, and provide methods
#   for obtaining data samples to be feed into tensor flow
##########################################################
import numpy as np
import hotvector as hv
import pickle
import scipy.sparse as sparse
import sys

class TupleData:
    'Class that stores the 3 hot vectors for tuple data and provides operations for using the data'

    def __init__(self, in_data, window_size = 1):
        """
            in_data should be a n row matrix, where n = number of tuples,
            each row corresponds to tuple (as a 3 hot vector)
            For efficency, in_data needs to be a list of csr_matrix arrays (sparse arrays)
        """
        self.tuples = np.array(in_data)
        self.window_size = window_size #context window, how many to look at from both left and right
        self.windows = self.tuples.shape[0] - (2*self.window_size) #how many windows of size 4 we are allowed 
        self.current_sample = 0

    def get_sample(self, index, batch_size):
        """
            Get a ith (index) training batch from the data, this includes 
            a the two surronding tuples and the center tuple 
            as the target.
            Convert the sparse arrays to regular numpy arrays
            Return tuple 
            (context1, context2, target)
        """
        index = index*batch_size
        target_index = index + (self.window_size) #the index of the target tuple
        context1 = self.tuples[target_index - self.window_size].toarray()
        context2 = self.tuples[target_index + self.window_size].toarray()
        target = self.tuples[target_index].toarray()

        for i in range(batch_size-1): #get the rest in the batch
            target_index += 1 
            next_context1 = self.tuples[target_index - self.window_size].toarray()
            next_context2 = self.tuples[target_index + self.window_size].toarray()
            next_target = self.tuples[target_index].toarray()
            context1 = np.concatenate((context1, next_context1))
            context2 = np.concatenate((context2, next_context2))
            target = np.concatenate((target, next_target))

            
        
        return (context1, context2, target)

def get_tuple_data(filename, word_dict_file, rel_dict_file):
    """
        Return a TupleData object using the tuple data in filename
        and the dictionary maps specified by word_dict_file and rel_dict_file
    """
    word_dict = pickle.load( open(word_dict_file, "rb" ) )
#    print(word_dict)
    rel_dict = pickle.load( open(rel_dict_file, "rb" ) )
    data = load_tuples(filename, word_dict, rel_dict)
    return TupleData(data)


def load_tuples(filename, word_dict, rel_dict):
    """
        Read in the tuple data, convert 
        it to 3 hot vector, and return a list of numpy arrays
    """
    data = []

    REL_WORD_LIMIT = 5
    with open(filename) as f:
        for line in f:
            if line and not line.isspace():
                split_line = line.split("|")
                rel = split_line[4]
                rel = rel.strip().lower()
                #if len(rel.split()) <= REL_WORD_LIMIT: #don't use relations that are this long
                if rel in rel_dict and len(rel.split()) <= REL_WORD_LIMIT: #don't use relations that are this long or arnt in rel_dict (too uncommon)
                    vect = hv.convert_to_3hot(line, word_dict, rel_dict)
                    data.append(vect)
                else:
                    print(rel)
    return data


if __name__=="__main__":
    tupfile = sys.argv[1]
    word_dict = 'word.pkl'
    rel_dict = 'rel.pkl'
    data = get_tuple_data(tupfile, word_dict, rel_dict)
    pickle.dump(data, open("TupleDataOut.pkl", 'wb'))
