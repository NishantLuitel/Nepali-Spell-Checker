from indic_transliteration import sanscript
import pickle
import textdistance
from metaphone import doublemetaphone
import os

module_dir = os.path.dirname(__file__)
file_path1 = os.path.join(module_dir, 'data','trie_depth.pickle')
with open(file_path1,'rb') as f:
    trie_depth = pickle.load(f)
 
file_path2 = os.path.join(module_dir, 'data','depth_dict.pickle')
with open(file_path2,'rb') as f:
    depth_dict = pickle.load(f)


def check_distance(w, depth = 1,edit_distance = 2,candidates = []):
    '''
    
    
    '''
    count = 0
    #candidates = []
    words = depth_dict[depth]
    for word in words:
        if(textdistance.levenshtein.distance(w, word)) <= edit_distance:
            count+=1
            candidates.append(word)
    return (candidates,count)



def check_distance2(w, depth = 1,edit_distance = 2):
    '''
    
    
    
    '''
    words = depth_dict[depth]
    candidates = list(filter(lambda word:textdistance.levenshtein.distance(w, word) <= edit_distance,words))        
    return (candidates,len(candidates))

def phonetic_distance(word,word_list,top = 5,force = False,include_metaphone = False):
    english_text = sanscript.transliterate(word, sanscript.DEVANAGARI, sanscript.ITRANS)
    m = []
    m1 = doublemetaphone(english_text)
    for w in word_list:
        english_text2 = sanscript.transliterate(w, sanscript.DEVANAGARI, sanscript.ITRANS)    
        
        if include_metaphone!=True:
            m.append(textdistance.levenshtein.distance(english_text.lower(),english_text2.lower()))
        else:
            m2 = doublemetaphone(english_text2)
            m.append(textdistance.levenshtein.distance(m1,m2))
            
        #m.append(textdistance.sorensen_dice(english_text.lower(),english_text2.lower()))        
    sorted_list = list(sorted(zip(m,word_list)))
    top_list = [x for _,x in sorted_list]
    if len(top_list)<=top:
        return_list = top_list
    else:
        top_dis = sorted_list[top-1][0]
        return_list = [x for _,x in sorted_list if _<=top_dis ]
        if force == True:
            return return_list[:5]
    return return_list
    
    
def check_distance_trie(w, depth = 1,edit_distance = 2,candidates = []):
    '''
    
    
    '''
    candidates = list(trie_depth[depth].search(w,edit_distance))
    return (candidates,len(candidates))

def candidate_words(word,minimum = 1,start_depth = 0):
    '''
    
    '''
    c = None
    #If word length is less than 3 than only use edit distance of 1 or less
    if len(word)<=3:
        c,c_ = check_distance(word,depth = start_depth,edit_distance = 1,candidates = [])
        for i in range(len(depth_dict)-1):
            if c_ < minimum:
                c,c_ = check_distance(word, depth = start_depth+i+1,edit_distance = 1,candidates = c)
                
    #If word length is more than 3 than use edit distance of 2 or less
    else:        
        c,c_ = check_distance(word, depth = start_depth,candidates = [])
        for i in range(len(depth_dict)-1):
            if c_ < minimum:
                #print("Entered depth, ",i+1)
                c,c_ = check_distance(word, depth = start_depth+i+1,candidates = c)
    
    #Filter 2 edit of type delete
    c = list(filter(lambda w: len(w)>=len(word)-1,c))
    if word not in c:
        c.append(word)
    return c


def candidate_words_trie(word,minimum = 1,start_depth = 0,edit_probabs = None):
    '''
    
    '''
    if edit_probabs == None:
        ed1 = 1
        ed2 = 2
    else:
        ed1 = edit_probabs[0]
        ed2 = edit_probabs[1]
    
    c = None
    #If word length is less than 3 than only use edit distance of 1 or less
    if len(word)<=3:
        c,c_ = check_distance_trie(word,depth = start_depth,edit_distance = ed1,candidates = [])
        for i in range(len(depth_dict)-1):
            if c_ < minimum:
                c,c_ = check_distance_trie(word, depth = start_depth+i+1,edit_distance = ed1,candidates = c)
                
    #If word length is more than 3 than use edit distance of 2 or less
    else:        
        c,c_ = check_distance_trie(word, depth = start_depth,edit_distance = ed2,candidates = [])
        for i in range(len(trie_depth)-1):
            if c_ < minimum:
                #print("Entered depth, ",i+1)
                c,c_ = check_distance_trie(word, depth = start_depth+i+1,edit_distance = ed2,candidates = c)
    
    #Filter 2 edit of type delete
    c = list(filter(lambda w: len(w)>=len(word)-1,c))
    if word not in c:
        c.append(word)
    return c


def final_candidate_words(word,minimum =1,top = 5,start_depth =0,force = False ,use_trie = False,time = False):
    
    if time:
        import time
        s = time.time()
    
    c = candidate_words(word,minimum = minimum,start_depth = start_depth) if use_trie == False else candidate_words_trie(word,minimum = minimum,start_depth = start_depth)
      
    
    if len(c) <6:
        if time:
            e = time.time()
            print("time passed fc-: ",e-s)
            print(c)
        return c
    else:
        if time:
            e = time.time()
            print("time passed fc: ",e-s)
            print(phonetic_distance(word,c,top = top))
        return phonetic_distance(word,c,top = top,force = force)
    
def return_lexicon_dict(huge=False): 


    with open(os.path.join(module_dir, 'data','name_with_additive.pk'),'rb') as f:
        name_with_additive = pickle.load(f)

    with open(os.path.join(module_dir, 'data','all_token2.pk'),'rb') as f:
        all_token2 = pickle.load(f)

    with open(os.path.join(module_dir, 'data','lexicon$.pk'),'rb') as f:
        lexicon_dict = pickle.load(f)
    if not huge:
        final_lexicon_dict = list(set(all_token2 + lexicon_dict))
    else:
        final_lexicon_dict = list(set(name_with_additive + all_token2 + lexicon_dict))
        
    return final_lexicon_dict