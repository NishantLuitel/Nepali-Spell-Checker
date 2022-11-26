
import regex as re
# from itertools import islice,izip
from collections import Counter
import itertools
import math
import pickle
import numpy


sample_factor = 0.05
alpha = 0.8
# def words(text): return re.findall(r'\w+', text.lower())


def words(text): return re.findall(r'[\u0900-\u097F]+', text.lower())


def words_bigram(text): return [tuple(x.split()) for x in re.findall
                                (r'\b[\u0900-\u097F]+\s[\u0900-\u097F]+',
                                 text.lower(), overlapped=True)]


def words_trigram(text): return re.findall(
    r'\b[\u0900-\u097F]+\s[\u0900-\u097F]+\s[\u0900-\u097F]+', text.lower(),
    overlapped=True)

#if __name__=='__main__':
    #WORDS = Counter(words(open('sample_np.txt', encoding="utf-8").read()))
with open('data/saved_words.pickle','rb') as inputfile:
    WORDS = pickle.load(inputfile) 
print(int(sample_factor*len(WORDS)))
    
WORDS = Counter(dict(WORDS.most_common(int(sample_factor*len(WORDS)))))
    
    
    
    #WORDS_bigram = Counter(words_bigram(
     #   open('sample_np.txt', encoding="utf-8").read()))
     
with open('data/saved_bi_words.pickle','rb') as inputfile:
    WORDS_bigram = pickle.load(inputfile) 
with open('data/saved_tri_words.pickle','rb') as inputfile:
    WORDS_trigram = pickle.load(inputfile) 
    #WORDS_trigram = Counter(words_trigram(
    #    open('data/sample_np.txt', encoding="utf-8").read()))

# List of all Nepali characters
char_vocab = []
for _ in range(2304, 2432):
    char_vocab += [chr(_)]


def probability(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return (WORDS[word] + 1)/ N


words_list = list(WORDS)
def likelihood(sentence,N=len(words_list)):
    prod = 1    
    for word in sentence:
        if word not in WORDS:
            prod*= 0.95
        else:
            word_index = words_list.index(word)
            proportional_word = words_list[-word_index+N-1]
            prod*= 0.05*probability(proportional_word)
    return prod

def likelihood2(sentence,candidate_sentence,candidate_count):
    prod = 1    
    i = 0
    #print(sentence.split(),candidate_sentence)
    for word,candidate_word in zip(sentence.split(),candidate_sentence):        
        if word==candidate_word:
            prod*= alpha
        else:
            N = candidate_count[i]
            prod*= (1-alpha)/N
        i+=1
    return prod


def probability_bigram(bi_word, N=sum(WORDS_bigram.values())):
    "Probability of `two words` given as a tuple."
    return (WORDS_bigram[bi_word]+1) / N

def probability_trigram(tri_word, N=sum(WORDS_trigram.values())):
    "Probability of `two words` given as a tuple."
    return (WORDS_trigram[tri_word]+1) / N

def posterior(word,context,p_lambda=1, prior = 'bigram'):
    if prior == 'bigram':
        return likelihood(word) * (probability_bigram(context)**p_lambda)
    
    if prior == 'trigram':
        return likelihood(word) * probability_trigram(context)**p_lambda

def edits1(word):
    "All edits that are one edit away from `word`."
    letters = char_vocab
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    "All edits that are two edits away from `word`."
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))

# Isn't exact


def edits3(word):
    "All edits that are two edits away from `word`."
    return set(e3 for e2 in known(edits2(word)) for e3 in edits1(e2))


def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)


def candidates_ordered(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])


def candidates_all(word):
    "Generate possible spelling corrections for word."
    return (set.union(known([word]), known(edits1(word)), known(edits2(word)) ,[word]))

def candidates_all_within1(word):
    "Generate possible spelling corrections for word."
    return set.union(known([word]), known(edits1(word)),[word])


def correction(word):
    "Most probable spelling correction for word."
    return max(candidates_ordered(word), key=probability)


def correctize(sentence, prior='bigram'):
    "Corrects the given 'sentence' using minimum edit"
    tokens = words(sentence)
    candidates = []    
    for _ in tokens:
        candidates.append(list(candidates_all_within1(_)))
    candidate_sentences = list(itertools.product(*candidates))
    #candidate_count = [len(_) for _ in candidate_sentences]
    
    if prior == 'trigram':
        #bigram tokens for possible sentences
        tri_tokens = [words_trigram(' '.join(sentence)) for sentence in candidate_sentences]
        tri_token_probab = []

        for row in tri_tokens:
            tri_token_probab.append([probability_trigram(_) for _ in row])
            
        sentence_likelihood = likelihood(sentence)
        #sentences_probab = [math.prod(row) for row in tri_token_probab]
        sentences_probab_post = [math.prod(row)*sentence_likelihood for row in tri_token_probab]
        # sorted_index = numpy.argsort(sentences_probab)
        
        sorted_index = numpy.argsort(sentences_probab_post)
        sentences_probab_post_sorted = sorted(sentences_probab_post,reverse = True)
    #return candidate_sentences[sentences_probab.index(max(sentences_probab))]
        return [candidate_sentences[k] for k in sorted_index[::-1]],sentences_probab_post_sorted
    
    if prior == 'bigram':
        #bigram tokens for possible sentences
        bi_tokens = [words_bigram(' '.join(sentence)) for sentence in candidate_sentences]
        bi_token_probab = []
        for row in bi_tokens:
            bi_token_probab.append([probability_bigram(_) for _ in row])  
            #sentence_likelihood_ = likelihood2(sentence,candidate_sentences)
            
        sentence_likelihood = likelihood(sentence)
        
        sentences_probab_post = [math.prod(row)*sentence_likelihood for row in bi_token_probab]
        #sentences_probab_post = [math.prod(row)*likelihood2(sentence,candidate_sentence,candidate_count) for row,candidate_sentence in zip(bi_token_probab,candidate_sentences)]
        
        sorted_index = numpy.argsort(sentences_probab_post)
        sentences_probab_post_sorted = sorted(sentences_probab_post,reverse = True)
    #return candidate_sentences[sentences_probab.index(max(sentences_probab))]
        return [candidate_sentences[k] for k in sorted_index[::-1]],sentences_probab_post_sorted

def correctize2(sentence, prior='bigram'):
    "Corrects the given 'sentence' using minimum edit"
    tokens = words(sentence)
    candidates = []    
    for _ in tokens:
        candidates.append(list(candidates_all_within1(_)))
    candidate_sentences = list(itertools.product(*candidates))
    candidate_count = [len(_) for _ in candidate_sentences]
    
    if prior == 'trigram':
        #bigram tokens for possible sentences
        tri_tokens = [words_trigram(' '.join(_)) for _ in candidate_sentences]
        tri_token_probab = []

        for row in tri_tokens:
            tri_token_probab.append([probability_trigram(_) for _ in row])
            
        #sentence_likelihood = likelihood(sentence)
        #sentences_probab = [math.prod(row) for row in tri_token_probab]
        sentences_probab_post = [math.prod(row)*likelihood2(sentence,candidate_sentence,candidate_count) for row,candidate_sentence in zip(tri_token_probab,candidate_sentences)]
        # sorted_index = numpy.argsort(sentences_probab)
        
        sorted_index = numpy.argsort(sentences_probab_post)
        sentences_probab_post_sorted = sorted(sentences_probab_post,reverse = True)
    #return candidate_sentences[sentences_probab.index(max(sentences_probab))]
        return [candidate_sentences[k] for k in sorted_index[::-1]],sentences_probab_post_sorted
    
    if prior == 'bigram':
        #bigram tokens for possible sentences
        bi_tokens = [words_bigram(' '.join(_)) for _ in candidate_sentences]
        bi_token_probab = []
        for row in bi_tokens:
            bi_token_probab.append([probability_bigram(_) for _ in row])  
            #sentence_likelihood_ = likelihood2(sentence,candidate_sentences)
            
        #sentence_likelihood = likelihood(sentence)
        
        # sentences_probab_post = [math.prod(row)*sentence_likelihood for row in bi_token_probab]
        sentences_probab_post = [math.prod(row)*likelihood2(sentence,candidate_sentence,candidate_count) for row,candidate_sentence in zip(bi_token_probab,candidate_sentences)]
        
        sorted_index = numpy.argsort(sentences_probab_post)
        sentences_probab_post_sorted = sorted(sentences_probab_post,reverse = True)
    #return candidate_sentences[sentences_probab.index(max(sentences_probab))]
        return [candidate_sentences[k] for k in sorted_index[::-1]],sentences_probab_post_sorted

 
    #return bi_token_probab

        
    
        
