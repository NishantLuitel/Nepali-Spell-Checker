from collections import Counter
import pickle


sample_factor = 0.01
sample_factor2 = 0.05

with open('data/saved_words.pickle','rb') as inputfile:
    WORDS = pickle.load(inputfile) 
print(int(sample_factor*len(WORDS)))

WORDS_full = WORDS
WORDS2 = Counter(dict(WORDS.most_common(int(sample_factor2*len(WORDS)))))
WORDS = Counter(dict(WORDS.most_common(int(sample_factor*len(WORDS)))))

    
with open('data/saved_bi_words.pickle','rb') as inputfile:
    WORDS_bigram = pickle.load(inputfile) 
with open('data/saved_tri_words.pickle','rb') as inputfile:
    WORDS_trigram = pickle.load(inputfile) 


 