"""
Sabastian Mugazambi and Simon Orlovsky
lsh.py
"""

import csv
import random
import numpy as np
import heapq
import math
import collections
import time

num_words = 0

def load(filename, number_of_lines):
    """ Loading and parsing files into a list sets """
    number_of_lines += 1
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        currentDocument = 0
        documents = []
        counter = 0

        chars = ""
        for row in reader:
            if len(row) == 1:
                chars += row[0] + " "
            else:
                if currentDocument != int(row[0]):
                    currentDocument = int(row[0])
                    documents.append(set())
                    counter += 1
                documents[-1].add(int(row[1]))
                if counter == (number_of_lines):
                    break

        documents.pop()
        chars = chars.split(" ")
        global num_words
        num_words = int(chars[1])
        return documents

def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def coprime(a, b):
    return gcd(a, b) == 1

def generate_hash(w):
    """ Each time we run a hash we need to generate a unique function """
    b = random.randint(0, w-1)
    found = False
    while not found:
        a = random.randint(0, w-1)
	if coprime(a,w):
            found = True
    return lambda x: (a*x+b)%w

def jaccard(documents, doc1 , doc2):
    """ Given the list of documents calculating the Jaccard value """
    doc1 = documents[doc1-1]
    doc2 = documents[doc2-1]

    inter = float(len(doc1.intersection(doc2)))
    union = float(len(doc1.union(doc2)))

    jacrd = inter/union
    return jacrd

def vote(neighbors, k, doc, sig_matrix):
    """ Deciding which neighbors to use for the knn """

    counter = collections.Counter(neighbors).most_common(k)
    num_docs = len(sig_matrix[0])-1
    k_nearest = []

    while len(counter) >= 1:
        for i in range(counter[0][1]):
            k_nearest.append(counter[0][0])
        counter.pop(0)


    while (len(k_nearest) < k):
        random_int = random.randint(0, num_docs)
        if random_int not in k_nearest:
             k_nearest.append(random_int)

    jaccards = []
    if len(k_nearest) > 0:
        for neighbor in k_nearest:
            jaccards.append(estimated_jac(sig_matrix,(neighbor+1), doc))
    else:
        jaccards.append(0)

    return np.mean(jaccards)

def brute_force(documents,doc,k):
    """ Brute force implementation of knn """
    #Make queue for inputting the jaccard values
    queue = []

    #calculate actual Jaccard
    for i in range (len(documents)):
        if i+1 == doc:
            pass
        else:
            jac = jaccard(documents,doc, i+1)
            heapq.heappush(queue, jac)

    #grab the largest k neigbors
    avg_largest = np.mean(heapq.nlargest(k,queue))
    return avg_largest

def make_char_mat(documents, number_of_words):
    """Return list of words with which documents contain them"""
    char_mat = []
    for i in range(number_of_words):
        row = set()
        for j in range(len(documents)):
            if i in documents[j]:
                row.add(j)
        if len(row) > 0:
            char_mat.append(row)
    return char_mat

def estimated_jac(sig_matrix, doc1, doc2):
    """ Find estimated jaccard values based on the signature matrix """
    doc1 -= 1
    doc2 -= 1

    sum = 0
    for row in sig_matrix:
        if row[doc1] == row[doc2]:
            sum += 1
    return float(sum)/len(sig_matrix)

def make_hash_cols(rows, num_words):
    """ Generate list of hash values corresponding to rows """
    row_hashes = []
    for i in range(rows):
        hash = generate_hash(num_words)
        hashes = []
        for j in range(num_words):
            hashes.append(hash(j))
        row_hashes.append(hashes)
    return np.array(row_hashes)

def make_sig_mat(char_mat, row_hashes, docs):
    """Return list of hashes with associated """
    sig_matrix = [[float('inf') for i in range(len(docs))] for i in range(len(row_hashes))]
    sig_matrix = np.array(sig_matrix)

    for i in range(len(sig_matrix)):
        for j in range(len(char_mat)):
            docs_with_word_j = char_mat[j]
            for doc in docs_with_word_j:
                if row_hashes[i][j] < sig_matrix[i][doc]:
                    sig_matrix[i][doc] = row_hashes[i][j]

    return sig_matrix

def make_bands(sig_matrix, r):
    """ convert signature matrix to dictionary of bands """
    sig_matrix = np.rot90(sig_matrix, 3)
    bands = math.ceil(len(sig_matrix[0])/float(r))
    sig_matrix = sig_matrix.tolist()
    band_dict = collections.defaultdict(list)

    for j in range(len(sig_matrix)):
        doc = sig_matrix[j]
        while (len(sig_matrix[j]) > 0):
            if len(sig_matrix[j]) < r:
                band_dict[tuple(doc[:])].append(j)
                doc = []
                sig_matrix[j] = []
                break
            else:
                band_dict[tuple(doc[:r])].append(j)
                doc = doc[r:]
                sig_matrix[j] = sig_matrix[j][r:]
                break

    return band_dict

def lsh_k(k, sig_matrix, doc, r):
    """ Implentation on LSH for KNN """

    banded_sig = make_bands(sig_matrix, r)
    cand_docs = []

    for key,value in banded_sig.iteritems():
        if (doc-1) in value:
            for val in value:
                if val != (doc-1):
                    cand_docs.append(val)

    return vote(cand_docs, k, doc, sig_matrix)

def main():
    """ Main function and user interface of this program """

    #Loading the files and setting parameters
    number_of_docs = int(raw_input("Number of documents: "))
    docs = load('docword.enron.txt', number_of_docs)

    print "\n*** Test calculating jaccard for two documents ***"
    docs_ids = raw_input("Please enter the two document ids in the following format. <id1> <id2>: ")
    #Creating the row hashes
    docs_ids = map(int, docs_ids.split(" "))
    doc1 = docs_ids[0]
    doc2 = docs_ids[1]
    print "Jaccard similarity for documents",doc1,"&",doc2,"is: ",jaccard(docs, doc1 , doc2)

    #Making the Characteristic Matrix
    print "\n*** Making characteristic matrix....***"
    char_mat = make_char_mat(docs, num_words)

    print "\n*** Making signature matrix....***"
    rows = int(raw_input("How many hash rows should I use: "))
    #Creating the row hashes
    row_hashes = make_hash_cols(rows, num_words)
    sig_matrix = make_sig_mat(char_mat, row_hashes,docs)

    print "\n*** Test calculating estimated jaccard for two documents ***"
    docs_ids_1 = raw_input("Please enter the two document ids in the following format. <id1> <id2>: ")
    docs_ids_1 = map(int, docs_ids_1.split(" "))
    doc_1 = docs_ids_1[0]
    doc_2 = docs_ids_1[1]
    
    print "Estimated Jaccard similarity for documents",doc_1,"&",doc_2,"is: ",estimated_jac(sig_matrix, doc1, doc2)

    print "\n*** Test Brute - force averages of k nearest neighbors***"
    # Brute - force Averages calculated here
    k = int(raw_input("Enter k for nearest neighbors averages: "))
    print "\n*** Finding jaccard averages for k by brute force........***"
    brute_avgs = []
    t0 = time.clock()
    for i in range (len(docs)):
        brute_avgs.append(brute_force(docs,i+1, k))
    brute_time = time.clock() - t0
    print "Average of averages for jaccard by bruteforce is: ",np.mean(brute_avgs)
    print "Brute force ran in: ",brute_time,"seconds"


    print "\n*** Test lsh averages of k nearest neighbors***"
    band_rows = int(raw_input("Enter r for number of rows in each band: "))

    lsh_avgs = []
    t0 = time.clock()
    for i in range (len(docs)):
        lsh_avgs.append(lsh_k(k, sig_matrix, i+1, band_rows))
    lsh_time = time.clock() - t0
    print "Average of averages for jaccard by lsh is: ",np.mean(lsh_avgs)
    print "lsh ran in: ",lsh_time,"seconds"

if __name__ == "__main__":
    main()
