import pickle
import numpy as np
import scipy.sparse as sparse
import sys

Person_tag = "_type_person"
Time_tag = "_type_time"
# Number_tag = "_TYPE_NUMBER"
Oragnization_tag ="_type_organization"
Unknown_tag = "_UNK_"

def cleaning(ATypes, A):
    tup_types = ATypes.split(',')
    for types in tup_types:
        type_name = types.split(':')[0].strip().lower()
        if "person" == type_name:
            if any(x.isupper() for x in A):
                A = Person_tag
        if "time_unit" == type_name:
            A = Time_tag
        #if "number" in colon:
        #    A = Number_tag
        if "organization" == type_name:
            if any(x.isupper() for x in A):
                A = Oragnization_tag

    return A


def hotvec(tup, word_dict, rel_dict):
    A1 = tup[0]
    rel = tup[1]
    A2 = tup[2]

    A1_hot = np.zeros(len(word_dict.keys()), dtype=np.int8)
    A2_hot = np.zeros(len(word_dict.keys()), dtype=np.int8)
    rel_hot = np.zeros(len(rel_dict.keys()), dtype=np.int8)

    pos_A1 = word_dict[A1]
    A1_hot[pos_A1] = 1

    pos_rel = rel_dict[rel]
    rel_hot[pos_rel] = 1

    pos_A2 = word_dict[A2]
    A2_hot[pos_A2] = 1

    hot3 = np.append(A1_hot, np.append(rel_hot, A2_hot))
  #  return hot3
    return sparse.csr_matrix(np.array(hot3)) #convert this to a sparse matrix for storage reasons (need to make it rank 2 for this)



def normalize_tuple(rel_line):
    """
        Convert rel_line (string representation
        of a tuple) to its corresponding 
        arguments (A1, A2) and relation, return tuple (A1, rel, A2)
        Replace also person, time_unit, and organization
        type entities with corresponding tags
    """
    split_line = rel_line.split("|")
    A1 = split_line[1]
    A1Types = split_line[2]
    rel = split_line[4]
    A2 = split_line[6]
    A2Types = split_line[7]
    A1 = cleaning(A1Types,A1)
    A2 = cleaning(A2Types,A2)
    return (A1.strip().lower(), rel.strip().lower(), A2.strip().lower())



def convert_to_3hot(rel_line, word_dict, rel_dict):
    """
        Convert rel_line (a string representation 
        of a tuple) to a corresponding 3 hot vector,
        using maps, word_dict and rel_dict
    """
    tup = normalize_tuple(rel_line)
    return hotvec(tup, word_dict, rel_dict)
    


def get_maps(tuple_file, word_dict_file="word.pkl", rel_dict_file="rel.pkl"):
    """
        create the dictionary maps for 
        the tuples in tuple_file
    """
    vocab_size = 0
    relvocab_size = 0
    word_dict = {}
    rel_dict = {}

    REL_WORD_LIMIT = 5

    with open(tuple_file) as f:

        for line in f:
            if line and not line.isspace():
                split_line = line.split("|")
                A1 = split_line[1]
                A1 = A1.strip()
                A1Types = split_line[2]

                rel = split_line[4]
                rel = rel.strip().lower()
                
                if len(rel.split()) <= REL_WORD_LIMIT: #don't use relations that are this long

                    A2 = split_line[6]
                    A2 = A2.strip()
                    A2Types = split_line[7]
        #            print(A2)
                    A1 = cleaning(A1Types,A1)
                    A2 = cleaning(A2Types,A2)

                    A1 = A1.lower()
                    A2 = A2.lower()

            #        print(A2)
                    # print(A1)
                    if(not A1 in word_dict):
                        word_dict[A1] = vocab_size
                        vocab_size += 1
                    if(not A2 in word_dict):
                        word_dict[A2] = vocab_size
                        vocab_size += 1
                    if(not rel in rel_dict):
                        rel_dict[rel] = relvocab_size
                        relvocab_size += 1
                else:
                    print(rel)

#    print(word_dict)
#    print("---------")
#    print(rel_dict)
    pickle.dump( word_dict, open( word_dict_file, "wb" ) )
    pickle.dump( rel_dict, open( rel_dict_file, "wb" ) )

def get_maps_filter(tuple_file, rel_thresh, word_thresh=0, word_dict_file="word.pkl", rel_dict_file="rel.pkl"):

    """
        create the dictionary maps for 
        the tuples in tuple_file
        Only use relations repeated in the data for rel_thresh times
        Only use words repeated in the data for word_thresh times
    """
    vocab_size = 0
    relvocab_size = 0
    word_dict = {}
    rel_dict = {}
    rel_counts = {}
#    word_counts = {}
    
    REL_WORD_LIMIT = 5

    with open(tuple_file) as f:

        for line in f:
            if line and not line.isspace():
                split_line = line.split("|")
                A1 = split_line[1]
                A1 = A1.strip()
                A1Types = split_line[2]

                rel = split_line[4]
                rel = rel.strip().lower()
                
                A2 = split_line[6]
                A2 = A2.strip()
                A2Types = split_line[7]

                 
                
                if len(rel.split()) <= REL_WORD_LIMIT: #don't use relations that are this long, dont use there argument either, they are also probably noise

                    #replace names of orgs and people with tags
                    A1 = cleaning(A1Types,A1)
   #                 print(A2)
                    A2 = cleaning(A2Types,A2)

   #                 print(A2)
   #                 print("------")
                    A1 = A1.lower()
                    A2 = A2.lower()


                    #add to dictonary 
                    if not A1 in word_dict:
                        word_dict[A1] = vocab_size
                        vocab_size += 1
                    if not A2 in word_dict:
                        word_dict[A2] = vocab_size
                        vocab_size += 1

                    #if(not rel in rel_dict):
                    #    rel_dict[rel] = relvocab_size
                    #    relvocab_size += 1

                    if not rel in rel_dict:
                        rel_dict[rel] = 1


                    #increment count dicts
                    #only count words and relations that appear in tuples where the relation is less than REL_WORD_LIMIT

                    #if(not A1 in word_counts):
                    #    word_counts[A1] = 1
                    #else:
                    #    word_counts[A1] = word_counts[A1] + 1

                    #if(not A2 in word_counts):
                    #    word_counts[A2] = 1
                    #else:
                    #    word_counts[A2] = word_counts[A2] + 1

                    if not rel in rel_counts:
                        rel_counts[rel] = 1
                    else:
                        rel_counts[rel] = rel_counts[rel] + 1


                else:
                    print(rel)

#    print(word_dict)
#    print("---------")
#    print(rel_dict)


    #Delete all keys that dont appear enough times in the data
#    for word in word_counts.keys():
#        if word_counts[word] < word_thresh:
#            del word_dict[word]

    for rel in rel_counts.keys():
        if rel_counts[rel] < rel_thresh:
            del rel_dict[rel]

    #assign indexes to the remaining relations in rel_dict
    index = 0
    for rel in rel_dict.keys():
        rel_dict[rel] = index
        index = index + 1

         

    pickle.dump( word_dict, open( word_dict_file, "wb" ) )
    pickle.dump( rel_dict, open( rel_dict_file, "wb" ) )


if __name__=="__main__":
    tupfile = sys.argv[1]
    REL_THRESH = int(sys.argv[2]) #only use relationships that appear at least 5 times in the data
#    get_maps(tupfile)
    get_maps_filter(tupfile, REL_THRESH)
    
