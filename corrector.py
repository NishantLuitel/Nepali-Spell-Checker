import pickle
from utils import final_candidate_words
import regex as re
import math
import itertools
import numpy



def words(text): 
    text = re.sub(r'[\u0964]', r'\u0020\u0964\u0020', text)
    return re.findall(r'[\u0900-\u097F]+', text.lower())

with open('models/bma.pickle','rb') as f:
    bma = pickle.load(f)

def words_bigram(text):   
    text = re.sub(r'[\u0964]', r'\u0020\u0964\u0020', text)
    return [tuple(x.split()) for x in re.findall
                                (r'\b[\u0900-\u097F]+\s[\u0900-\u097F]+',text.lower(), overlapped=True)]
    
    
def logprob(ngram,model,minimum):
    '''
    Calculate log probability
    
    
    '''    
    if ngram in model.lm[0]:
        return model.lm[0][ngram]
    return minimum
    
    
def likelihood_bm(sentence,candidate_sentence):
    '''
    Returns P(Possible Typo Sentence/Candidate Correct Sentence)
    
    Uses Naive approach to compute probability for sentence from individual words
    
    '''    
    
    prod = 1
    for word,candidate_word in zip(sentence.split(),candidate_sentence):          
        prod*= bma.likelihood(word,candidate_word)
    return prod


alpha = 0.65
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

def correctize_entire_knlm(sentence, model,p_lambda = 1,prior='bigram',trie = False,likelihood = 'default'):
    "Corrects the given 'sentence' using minimum edit"

    tokens = words(sentence)

    candidates = []    
    for _ in tokens:
        candidates.append(final_candidate_words(_,use_trie = trie))
    
   
    candidate_sentences = list(itertools.product(*candidates))
    minimum = min(model.lm[0].values())
    
    if prior == 'trigram':
        pass
    
    if prior == 'bigram':
        bi_tokens = [words_bigram(' '.join(_)) for _ in candidate_sentences]
        bi_token_probab = []
   
        for row in bi_tokens:
            bi_token_probab.append([logprob(tuple(_),model,minimum) for _ in row])  

        if likelihood=='default':
            candidate_count = [len(_) for _ in candidates]  
            sentences_probab_post=[(sum(row)*p_lambda) +
                                   math.log(constant_distributive_likelihood(sentence,candidate_sentence,candidate_count)) 
                                   for row,candidate_sentence in zip(bi_token_probab,candidate_sentences)]
        elif likelihood=='bm':
            sentences_probab_post=[(sum(row)*p_lambda) + 
                                    math.log(likelihood_bm(sentence,candidate_sentence)) 
                                    for row,candidate_sentence in zip(bi_token_probab,candidate_sentences)]


        
        sorted_index = numpy.argsort(sentences_probab_post)
        sentences_probab_post_sorted = sorted(sentences_probab_post,reverse = True)
        
        return [candidate_sentences[k] for k in sorted_index[::-1]],sentences_probab_post_sorted
    

def correctize_with_window_knlm(sentence,model,window = 5,p_lambda = 1,prior = 'bigram',trie = False,likelihood = 'default'):
    '''
    
    '''   
    
    tokens = words(sentence)
    if len(tokens) <= window:
        return correctize_entire_knlm(sentence,model,p_lambda=p_lambda,prior = prior,trie = trie,likelihood = likelihood)
    else:
        windows = [tokens[n:window+n] for n in range(0,len(tokens),window-1) if window+n <len(tokens)-1]    
        remaining = (window-1)*len(windows)
        windows.append(tokens[remaining:])
        corrects = []
        for _ in windows:
            d = correctize_entire_knlm(' '.join(_),model,p_lambda=p_lambda,prior = prior,trie = trie,likelihood = likelihood)
            corrects.append(d)
        return corrects
    
    
def return_choices2(sample_sentences,model,p_lambda = 1,trie = False,likelihood = 'default'):
    d = correctize_with_window_knlm(sample_sentences,model,p_lambda =p_lambda,trie = trie,likelihood = likelihood)
    window_candidates = []
    window_probab = []
    for window in d:
        maxim = min(len(window[0]),10)
        top_candidates = window[0][:maxim]
        window_candidates.append(top_candidates)
        window_probab.append(window[1][:maxim])
    return window_candidates,window_probab
        
    
def extract_choices(sample_sentences,model,p_lambda = 1,trie = False,likelihood = 'default'):
    
    wc,wp = return_choices2(sample_sentences,model,p_lambda = p_lambda,trie = trie ,likelihood = likelihood)
#     choices_list=[set() for i in range(len(sample_sentences.split())+1)]
    choices_list=[[] for i in range(len(sample_sentences.split())+1)]
    print(len(choices_list))

    const = 0
    for _ in wc:
        for sens in _:
            for i,w in enumerate(sens):
                index = i + const
                if w not in choices_list[index]:
                    choices_list[index].append(w)
        const += len(wc[0][0])-1
    return choices_list