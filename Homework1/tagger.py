import sys
import nltk
import math
import time
import collections
import numpy as np

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    for line in brown_train:
        # sentences
        tmp_words = []
        tmp_tags = []
        tmpline = line.strip().split()
        tmp_words = tmp_words + [START_SYMBOL, START_SYMBOL]
        tmp_tags = tmp_tags + [START_SYMBOL, START_SYMBOL]
        for pair in tmpline:
            split_vec = pair.split("/")
            tmp_len = len(split_vec)
            if tmp_len == 2:
                tmp_words.append(split_vec[0])
                tmp_tags.append(split_vec[1])
            else:
                tmp_words.append("/".join(split_vec[:tmp_len]))
                tmp_tags.append(split_vec[tmp_len-1])
        tmp_words.append(STOP_SYMBOL)
        tmp_tags.append(STOP_SYMBOL)
        brown_words.append(tmp_words)
        brown_tags.append(tmp_tags)
    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    bigram_count = collections.defaultdict(int)    
    trigram_count = collections.defaultdict(int)
    for line in brown_tags:
        # bigram
        for gram in nltk.bigrams(line):
            bigram_count[gram] += 1
        # trigram
        for gram in nltk.trigrams(line):
            trigram_count[gram] += 1
        # prob calculation
        for gram in trigram_count.keys():
            q_values[gram] = math.log(float(trigram_count[gram])/float(bigram_count[gram[:2]]), 2)
    return q_values

#TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates the tag trigrams in reverse.  In other words, instead of looking at the probabilities that the third tag follows the first two, look at the probabilities of the first tag given the next two.
# Hint: This code should only differ slightly from calc_trigrams(brown_tags)
def calc_trigrams_reverse(brown_tags):
    q_values = {}
    bigram_count = collections.defaultdict(int)    
    trigram_count = collections.defaultdict(int)
    for line in brown_tags:
        line = line[::-1]
        # bigram
        for gram in nltk.bigrams(line):
            bigram_count[gram] += 1
        # trigram
        for gram in nltk.trigrams(line):
            trigram_count[gram] += 1
        # prob calculation
        for gram in trigram_count.keys():
            q_values[gram] = math.log(float(trigram_count[gram])/float(bigram_count[gram[:2]]), 2)
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    unigram_count = collections.defaultdict(int)
    known_words = set([])
    for line in brown_words:
        # bigram
        for gram in line:
            unigram_count[gram] += 1
    for word in unigram_count.keys():
        if unigram_count[word] > RARE_WORD_MAX_FREQ:
            known_words.add(word)
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    for i, line in enumerate(brown_words):
        for j, word in enumerate(line):
            if word not in known_words:
                brown_words[i][j] = RARE_SYMBOL
    brown_words_rare = brown_words
    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    word_tag_count = collections.defaultdict(int)
    tag_count = collections.defaultdict(int)
    taglist = set([])
    # collect the possible tags
    for i,line in enumerate(brown_tags):
        for j,tag in enumerate(line):
            if tag not in taglist:
                taglist.add(tag)
            tag_count[tag] += 1
            word_tag_count[(brown_words_rare[i][j],tag)] += 1
    # calculate the emission prob
    for i, pair in enumerate(word_tag_count):
        e_values[pair] = math.log(float(word_tag_count[pair])/float(tag_count[pair[1]]),2)

    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    
    taglist = list(taglist)
    taglist.sort()
    taglist = taglist[1:]
    len_tag = len(taglist)
    # loop for the line
    counter = 0
    
    
    for line in brown_dev_words:
        counter += 1
        if counter%100 == 0:
            print(counter)
        tmpline = [w if w in known_words else RARE_SYMBOL for w in line]
        len_obs = len(tmpline) 
        prob_matrix = collections.defaultdict(float)
        backpointer = collections.defaultdict(int)
        # initialization    
        for i, tag in enumerate(taglist):
            #print("qvalue")
            #print(q_values[(START_SYMBOL, START_SYMBOL, tag)] )
            #print("evalue")
            #print(e_values[(tmpline[0],tag)] )
            prob_matrix[(i,0)] = q_values.get((START_SYMBOL, START_SYMBOL, tag), LOG_PROB_OF_ZERO) + e_values.get((tmpline[0], tag), LOG_PROB_OF_ZERO) 
            backpointer[(i,0)] = -1
        
        # recursion
        if len_obs > 1:
            for i, tag_c in enumerate(taglist):
                tmp_max = -float('inf')
                prob_matrix[(i, 1)] = -float('inf')
                for j, tag_p in enumerate(taglist):
                    tmp_prob = prob_matrix[(j, 0)] + q_values.get((START_SYMBOL, tag_p, tag_c), LOG_PROB_OF_ZERO) + e_values.get((tmpline[1], tag_c), LOG_PROB_OF_ZERO)
                    if tmp_prob > tmp_max:
                        prob_matrix[(i,1)] = tmp_prob
                        backpointer[(i,1)] = j
                        tmp_max = tmp_prob
        if len_obs > 2:
            for t in range(2,len_obs):
                for i, tag_c in enumerate(taglist):
                    tmp_max = -float('inf')
                    prob_matrix[(i, t)] = tmp_max
                    for j, tag_p in enumerate(taglist):
                        tmp_prob = prob_matrix[(j, t-1)] + q_values.get((taglist[backpointer[(j,t-1)]], tag_p, tag_c), LOG_PROB_OF_ZERO) + e_values.get((tmpline[t], tag_c), LOG_PROB_OF_ZERO)
                        if tmp_prob > tmp_max:
                            prob_matrix[(i,t)] = tmp_prob
                            backpointer[(i,t)] = j
                            tmp_max = tmp_prob
                        
        # trace back
        tag_index = []
        tmp_max_index = 0
        for i in range(len_tag):
            if prob_matrix[(i,len_obs-1)] > prob_matrix[(tmp_max_index,len_obs-1)]:
                tmp_max_index = i

        end_index = tmp_max_index
        tag_index.append(end_index)
        for rt in range(len_obs-1,0,-1):
            end_index = backpointer[(end_index,rt)]
            tag_index.append(end_index)
        tag_index = tag_index[::-1]
        tmp_tagged = []
        for i in range(len_obs):
            item = "/".join([line[i],taglist[tag_index[i]]])
            tmp_tagged.append(item)
        tagged.append(" ".join(tmp_tagged) + "\n")           
    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    df_tagger = nltk.DefaultTagger("NOUN")
    bigram_tagger = nltk.BigramTagger(training, backoff=df_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
    for line in brown_dev_words:
        pair_list = ["/".join([word, tag]) for word, tag in trigram_tagger.tag(line)]
        tmpline = " ".join(pair_list) + "\n"
        tagged.append(tmpline)
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = '/home/classes/cs477/data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities with an option to reverse (question 7)
    if len(sys.argv) > 1 and sys.argv[1] == "-reverse":
        q_values = calc_trigrams_reverse(brown_tags)
    else:
        q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
