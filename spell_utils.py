import regex as re
from collections import Counter
import itertools
import math
import pickle
import numpy


sample_factor = 0.01
sample_factor2 = 0.05


alpha = 0.65

def words(text): 
    text = re.sub(r'[\u0964]', r'\u0020\u0964\u0020', text)
    return re.findall(r'[\u0900-\u097F]+', text.lower())


def words_bigram(text):   
    text = re.sub(r'[\u0964]', r'\u0020\u0964\u0020', text)
    return [tuple(x.split()) for x in re.findall
                                (r'\b[\u0900-\u097F]+\s[\u0900-\u097F]+',text.lower(), overlapped=True)]

def words_trigram(text):
    text = re.sub(r'[\u0964]', r'\u0020\u0964\u0020', text)
    return re.findall(
    r'\b[\u0900-\u097F]+\s[\u0900-\u097F]+\s[\u0900-\u097F]+', text.lower(),
    overlapped=True)


def words_bigram_from_list(sentence):
    return [(sentence[i],sentence[i+1]) for i in range(len(sentence)-1)]


# List of all Nepali characters
char_vocab = []
for _ in range(2304, 2432):
    char_vocab += [chr(_)]


def change_keys(d):
    return dict([(k[0], v) for k, v in d.items()])

with open('data/saved_words_counter1','rb') as inputfile:
    WORDS = pickle.load(inputfile) 
    WORDS = Counter(change_keys(WORDS))
    
    


with open('data/saved_words_counter2','rb') as inputfile:
    WORDS_bigram = pickle.load(inputfile) 
# with open('data/saved_words_counter3','rb') as inputfile:
#     WORDS_trigram = pickle.load(inputfile) 
#comment this code for trigrams
WORDS_trigram = WORDS_bigram


#Words Partition(splits)
WORDS_full = WORDS
WORDS2 = Counter(dict(WORDS.most_common(int(sample_factor2*len(WORDS)))))
WORDS = Counter(dict(WORDS.most_common(int(sample_factor*len(WORDS)))))




#Token Probability
def probability(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return (WORDS[word] + 1)/ N

def probability_bigram(bi_word, N=sum(WORDS_bigram.values())):
    "Probability of `two words` given as a tuple."
    return (WORDS_bigram[bi_word]+1) / N

def probability_trigram(tri_word, N=sum(WORDS_trigram.values())):
    "Probability of `two words` given as a tuple."
    return (WORDS_trigram[tri_word]+1) / N


words_list = list(WORDS)


#Likelihood models
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

def constant_distributive_likelihood(sentence,candidate_sentence,candidate_count):
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




# def posterior(word,context,p_lambda=1, prior = 'bigram'):
#     if prior == 'bigram':
#         return likelihood(word) * (probability_bigram(context)**p_lambda)
    
#     if prior == 'trigram':
#         return likelihood(word) * probability_trigram(context)**p_lambda

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



def edits2_(word):
    for e1 in known_from_full(edits1(word)):
        s = set( e2 for  e2 in edits1(e1))
        return known_from_full(s)
# Isn't exact


def edits3(word):
    "All edits that are two edits away from `word`."
    return set(e3 for e2 in known(edits2(word)) for e3 in edits1(e2))


def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def known_from_WORDS2(words):
    return set(w for w in words if (w,) in WORDS2 or w in WORDS2)
    

def known_from_full(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if (w,) in WORDS_full or w in WORDS_full)


def candidates_ordered(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])


def candidates_all(word):
    "Generate possible spelling corrections for word."
    return (set.union(known([word]), known(edits1(word)), known(edits2(word)) ,[word]))

def candidates_all_from_full(word):
    "Generate possible spelling corrections for word."
    return (set.union(known_from_full([word]), known(edits1(word)), edits2_(word) ,[word]))

def candidates_all_within1(word):
    
    "Generate possible spelling corrections for word."
    return set.union(known_from_full([word]), known(edits1(word)),[word])

def candidates_all_within1_full(word):
    
    "Generate possible spelling corrections for word."
    return set.union(known_from_full([word]), known(edits1(word)),[word])


def candidates_all_within1_full_expanded(word):
    
    "Generate possible spelling corrections for word."
   
    return set.union(known_from_full([word]), known_from_WORDS2(edits1(word)),[word])


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
        #trigram tokens for possible sentences
        tri_tokens = [words_trigram(' '.join(sentence)) for sentence in candidate_sentences]
        tri_token_probab = []

        for row in tri_tokens:
            tri_token_probab.append([probability_trigram(_) for _ in row])
            
        sentence_likelihood = likelihood(sentence)
        sentences_probab_post = [math.prod(row)*sentence_likelihood for row in tri_token_probab]
        sorted_index = numpy.argsort(sentences_probab_post)
        sentences_probab_post_sorted = sorted(sentences_probab_post,reverse = True)

        return [candidate_sentences[k] for k in sorted_index[::-1]],sentences_probab_post_sorted
    
    if prior == 'bigram':
        #bigram tokens for possible sentences
        bi_tokens = [words_bigram(' '.join(sentence)) for sentence in candidate_sentences]
        bi_token_probab = []
        for row in bi_tokens:
            bi_token_probab.append([probability_bigram(_) for _ in row])  
        sentence_likelihood = likelihood(sentence)
        
        sentences_probab_post = [math.prod(row)*sentence_likelihood for row in bi_token_probab]
        #sentences_probab_post = [math.prod(row)*likelihood2(sentence,candidate_sentence,candidate_count) for row,candidate_sentence in zip(bi_token_probab,candidate_sentences)]
        
        sorted_index = numpy.argsort(sentences_probab_post)
        sentences_probab_post_sorted = sorted(sentences_probab_post,reverse = True)
    #return candidate_sentences[sentences_probab.index(max(sentences_probab))]
        return [candidate_sentences[k] for k in sorted_index[::-1]],sentences_probab_post_sorted

def correctize_entire_with_time(sentence, p_lambda = 1,prior='bigram',tokenized = False):
    "Corrects the given 'sentence' using minimum edit"
    import time

    t_start = time.time()
    tokens = words(sentence)
    start1 = time.time()
    candidates = []    
    for _ in tokens:
        candidates.append(list(filter(lambda word: word in tokens or WORDS2[word]>1000 ,list(candidates_all_within1_full_expanded(_)))))
    candidate_count = [len(_) for _ in candidates]  
    print(candidate_count[0:len(candidates)])      
    end1 = time.time()
    print("Time passed", end1-start1,"sec")
    
    start1 = time.time()
    candidate_sentences = list(itertools.product(*candidates))
    end1 = time.time()
    print("Time passed", end1-start1,"sec")


    
    if prior == 'trigram':
        #bigram tokens for possible sentences
        tri_tokens = [words_trigram(' '.join(_)) for _ in candidate_sentences]
        tri_token_probab = []

        for row in tri_tokens:
            tri_token_probab.append([probability_trigram(_) for _ in row])
            
        #sentence_likelihood = likelihood(sentence)
        #sentences_probab = [math.prod(row) for row in tri_token_probab]
        sentences_probab_post = [math.prod(row)*constant_distributive_likelihood(sentence,candidate_sentence,candidate_count) for row,candidate_sentence in zip(tri_token_probab,candidate_sentences)]
        # sorted_index = numpy.argsort(sentences_probab)
        
        sorted_index = numpy.argsort(sentences_probab_post)
        sentences_probab_post_sorted = sorted(sentences_probab_post,reverse = True)
    #return candidate_sentences[sentences_probab.index(max(sentences_probab))]
        return [candidate_sentences[k] for k in sorted_index[::-1]],sentences_probab_post_sorted
    
    if prior == 'bigram':
        start1 = time.time()
        #bigram tokens for possible sentences
        bi_tokens = [words_bigram(' '.join(_)) for _ in candidate_sentences]
        #bi_tokens = [[a,b for zip(_[:-1],_[1:])] for _ in candidate_sentences]
        end1 = time.time()
        print("Time passed", end1-start1,"sec")
        
        bi_token_probab = []
        start1 = time.time()
        for row in bi_tokens:
            bi_token_probab.append([probability_bigram(_) for _ in row])  
            #sentence_likelihood_ = likelihood2(sentence,candidate_sentences)
        end1 = time.time()
        print("Time passed", end1-start1,"sec")
        #sentence_likelihood = likelihood(sentence)
        
        start1 = time.time()
        # sentences_probab_post = [math.prod(row)*sentence_likelihood for row in bi_token_probab]
        sentences_probab_post = [math.log((math.prod(row)**p_lambda)*constant_distributive_likelihood(sentence,candidate_sentence,candidate_count)) for row,candidate_sentence in zip(bi_token_probab,candidate_sentences)]
        #sentences_log_probab = [math.ln(m) for m in sentences_probab_post]
        end1 = time.time()
        print("Time passed", end1-start1,"sec")
        
        sorted_index = numpy.argsort(sentences_probab_post)
        sentences_probab_post_sorted = sorted(sentences_probab_post,reverse = True)
        

        t_end = time.time()
        print("Total Time passed", t_end-t_start,"sec")
    #return candidate_sentences[sentences_probab.index(max(sentences_probab))]
        return [candidate_sentences[k] for k in sorted_index[::-1]],sentences_probab_post_sorted
    

def correctize_entire(sentence, p_lambda = 1,prior='bigram',tokenized = False):
    "Corrects the given 'sentence' using minimum edit"

    tokens = words(sentence)

    candidates = []    
    for _ in tokens:
        candidates.append(list(filter(lambda word: word in tokens or WORDS2[word]>1000 ,list(candidates_all_within1_full_expanded(_)))))
    candidate_count = [len(_) for _ in candidates]  
   
    candidate_sentences = list(itertools.product(*candidates))



    
    if prior == 'trigram':
        #bigram tokens for possible sentences
        tri_tokens = [words_trigram(' '.join(_)) for _ in candidate_sentences]
        tri_token_probab = []

        for row in tri_tokens:
            tri_token_probab.append([probability_trigram(_) for _ in row])
            
        #sentence_likelihood = likelihood(sentence)
        #sentences_probab = [math.prod(row) for row in tri_token_probab]
        sentences_probab_post = [math.prod(row)*constant_distributive_likelihood(sentence,candidate_sentence,candidate_count) for row,candidate_sentence in zip(tri_token_probab,candidate_sentences)]
        # sorted_index = numpy.argsort(sentences_probab)
        
        sorted_index = numpy.argsort(sentences_probab_post)
        sentences_probab_post_sorted = sorted(sentences_probab_post,reverse = True)
    #return candidate_sentences[sentences_probab.index(max(sentences_probab))]
        return [candidate_sentences[k] for k in sorted_index[::-1]],sentences_probab_post_sorted
    
    if prior == 'bigram':

        #bigram tokens for possible sentences
        bi_tokens = [words_bigram(' '.join(_)) for _ in candidate_sentences]
        #bi_tokens = [[a,b for zip(_[:-1],_[1:])] for _ in candidate_sentences]

  
        
        bi_token_probab = []
   
        for row in bi_tokens:
            bi_token_probab.append([probability_bigram(_) for _ in row])  
            #sentence_likelihood_ = likelihood2(sentence,candidate_sentences)

        #sentence_likelihood = likelihood(sentence)

        # sentences_probab_post = [math.prod(row)*sentence_likelihood for row in bi_token_probab]
        sentences_probab_post = [math.log((math.prod(row)**p_lambda)*constant_distributive_likelihood(sentence,candidate_sentence,candidate_count)) for row,candidate_sentence in zip(bi_token_probab,candidate_sentences)]
        #sentences_log_probab = [math.ln(m) for m in sentences_probab_post]

        
        sorted_index = numpy.argsort(sentences_probab_post)
        sentences_probab_post_sorted = sorted(sentences_probab_post,reverse = True)
        

    #return candidate_sentences[sentences_probab.index(max(sentences_probab))]
        return [candidate_sentences[k] for k in sorted_index[::-1]],sentences_probab_post_sorted

def tupler(x):
        return tuple(x.split())
    
# def logprob(ngram,kn_lm2):
#     for i, order in enumerate(kn_lm2.lm):
#         if ngram[i:] in order:
#             return order[ngram[i:]]
#     #print(kn_lm2.lm[-1].values())
#     return min(kn_lm2.lm[0].values())



def logprob(ngram,kn_lm2,minimum):
    if ngram in kn_lm2.lm[0]:
        return kn_lm2.lm[0][ngram]
    return minimum

def score_sent(sent):
    """
    Return log prob of the sentence.

    Params:
        sent [tuple->string] The words in the unpadded sentence.
    """
    padded = (
        ('<s>',) * (2 - 1) + sent)
    sent_logprob = 0
    for i in range(len(sent) - 2 + 1):
        ngram = sent[i:i+2]
        sent_logprob += logprob(ngram)
    return sent_logprob

def correctize_entire_knlm(sentence, model,p_lambda = 1,prior='bigram',tokenized = False):
    "Corrects the given 'sentence' using minimum edit"

    tokens = words(sentence)

    candidates = []    
    for _ in tokens:
        candidates.append(list(filter(lambda word: word in tokens or WORDS2[word]>1000 ,list(candidates_all_within1_full_expanded(_)))))
    candidate_count = [len(_) for _ in candidates]  
   
    candidate_sentences = list(itertools.product(*candidates))


    minimum = min(model.lm[0].values())
    
    if prior == 'trigram':
        #bigram tokens for possible sentences
        tri_tokens = [words_trigram(' '.join(_)) for _ in candidate_sentences]
        tri_token_probab = []

        for row in tri_tokens:
            tri_token_probab.append([logprob(tuple(_),model,minumum) for _ in row])
            
        #sentence_likelihood = likelihood(sentence)
        #sentences_probab = [math.prod(row) for row in tri_token_probab]
        sentences_probab_post=[(sum(row)*p_lambda)+math.log(constant_distributive_likelihood(sentence,candidate_sentence,candidate_count)) for row,candidate_sentence in zip(tri_token_probab,candidate_sentences)]
        # sorted_index = numpy.argsort(sentences_probab)
        
        sorted_index = numpy.argsort(sentences_probab_post)
        sentences_probab_post_sorted = sorted(sentences_probab_post,reverse = True)
        return [candidate_sentences[k] for k in sorted_index[::-1]],sentences_probab_post_sorted
    
    if prior == 'bigram':

        #bigram tokens for possible sentences
        bi_tokens = [words_bigram(' '.join(_)) for _ in candidate_sentences]
        #bi_tokens = [[a,b for zip(_[:-1],_[1:])] for _ in candidate_sentences]
        bi_token_probab = []
   
        for row in bi_tokens:
            bi_token_probab.append([logprob(tuple(_),model,minimum) for _ in row])  
            #sentence_likelihood_ = likelihood2(sentence,candidate_sentences)

        #sentence_likelihood = likelihood(sentence)

        # sentences_probab_post = [math.prod(row)*sentence_likelihood for row in bi_token_probab]
        sentences_probab_post=[(sum(row)*p_lambda)+math.log(constant_distributive_likelihood(sentence,candidate_sentence,candidate_count)) for row,candidate_sentence in zip(bi_token_probab,candidate_sentences)]
        #sentences_log_probab = [math.ln(m) for m in sentences_probab_post]
        #print(sentences_probab_post)

        
        sorted_index = numpy.argsort(sentences_probab_post)
        sentences_probab_post_sorted = sorted(sentences_probab_post,reverse = True)
        

    #return candidate_sentences[sentences_probab.index(max(sentences_probab))]
        return [candidate_sentences[k] for k in sorted_index[::-1]],sentences_probab_post_sorted
    
def correctize_with_window(sentence,window = 5,p_lambda = 1,prior = 'bigram'):
    tokens = words(sentence)
    if len(tokens) <= window:
        return correctize_entire(sentence,p_lambda=p_lambda,prior = prior)
    else:
        windows = [tokens[n:window+n] for n in range(0,len(tokens),window-1) if window+n <len(tokens)-1]    
        remaining = (window-1)*len(windows)
        windows.append(tokens[remaining:])
        corrects = []
        for _ in windows:
            #corrects.append(correctize3(' '.join(_)))
            d = correctize_entire(' '.join(_),p_lambda=p_lambda,prior = prior)
            corrects.append(d)
        return corrects
    
def correctize_with_window_knlm(sentence,model,window = 5,p_lambda = 1,prior = 'bigram'):
    '''
        
    
    '''  
    tokens = words(sentence)
    if len(tokens) <= window:
        return correctize_entire_knlm(sentence,model,p_lambda=p_lambda,prior = prior)
    else:
        windows = [tokens[n:window+n] for n in range(0,len(tokens),window-1) if window+n <len(tokens)-1]    
        remaining = (window-1)*len(windows)
        windows.append(tokens[remaining:])
        corrects = []
        for _ in windows:
            #corrects.append(correctize3(' '.join(_)))
            d = correctize_entire_knlm(' '.join(_),model,p_lambda=p_lambda,prior = prior)
            corrects.append(d)
        return corrects
   

    
def return_choices(sample_sentences,model):
    d = correctize_with_window_knlm(sample_sentences,model)
    top_choice,other_choices = print_corrected_sentence(d)


    choices_list=[set() for i in range(len(sample_sentences.split())+1)]
    print(len(choices_list))
    #_  = [item for sublist in other_choices for item in sublist]
    #print(_)
    const = 0
    for _ in other_choices:
        for sens in _:
            for i,w in enumerate(sens):
                index = i + const
                choices_list[index].add(w)
        const += len(other_choices[0][0])-1

    return choices_list  

    
def print_corrected_sentence(d,j = 0):
    s = ''
    k = []
    if(len(d)>1):
        for i in range(len(d)-1):
            s += ' '.join(d[i][0][j][0:4])
            s+=' '
            k.append(d[i][0][0:5])
    s+=' '.join(d[len(d)-1][0][j])
    k.append(d[len(d)-1][0][0:5])
    return s,k
    #return bi_token_probab
    
    
    

def timer(fun,args):
    import time
    s = time.time()
    k = fun(args)
    e = time.time()
    print("Time taken, : ",e-s," sec")
    return k 
    
        

