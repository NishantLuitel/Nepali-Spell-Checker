import pickle
from BrillMoore import BrillMore

def gather_dataset(filename):
    '''
    Opens the filename containing dataset
    
    '''
    
    with open(filename,'rb') as f:
        dataset = pickle.load(f)
    assert type(dataset)==list, "Put the correctly generated dataset"
    return dataset


def WER(correct_tokens,predicted_tokens):
    '''
    Calculates the Word error rate for given list of correct and predicted tokens
    
    '''
    
    assert len(correct_tokens)==len(predicted_tokens),"Dimension of correct and predicted tokens don't match"
    
    if type(correct_tokens[0]) == list or type(correct_tokens[0]) == tuple:
        correct_tokens = [w for s in correct_tokens for w in s]
        predicted_tokens = [w for s in predicted_tokens for w in s]
    
    if type(correct_tokens[0]) == str:
        total = len(correct_tokens)
        errors = 0
        for c,p in zip(correct_tokens,predicted_tokens):
            if c!=p:
                errors+=1
#                 print(errors)
        return round(errors/total,3)

def word_accuracy(correct_tokens,predicted_tokens,error_tokens):
    '''
    Calculates the Word accuracy rate for given list of correct,predicted and error tokens
    
    '''
    
    assert len(correct_tokens)==len(predicted_tokens),"Dimension of correct and predicted tokens don't match"
    
    if type(correct_tokens[0]) == list or type(correct_tokens[0]) == tuple:
        correct_tokens = [w for s in correct_tokens for w in s]
        predicted_tokens = [w for s in predicted_tokens for w in s]
        error_tokens = [w for s in error_tokens for w in s]
    
    if type(correct_tokens[0]) == str:
        corrected = 0
        errors = 0
        for c,p,e in zip(correct_tokens,predicted_tokens,error_tokens):
            if c!=e:
                errors+=1
                if c==p:
                    corrected+=1
#                 print(errors)
        return (round(corrected/errors,3),corrected,errors)    

align = BrillMore().align
def char_accuracy(correct_tokens,predicted_tokens,error_tokens):
    '''
    Calculates the Word accuracy rate for given list of correct,predicted and error tokens
    
    '''
    
    assert len(correct_tokens)==len(predicted_tokens),"Dimension of correct and predicted tokens don't match"
    
    if type(correct_tokens[0]) == list or type(correct_tokens[0]) == tuple:
        correct_tokens = [w for s in correct_tokens for w in s]
        predicted_tokens = [w for s in predicted_tokens for w in s]
        error_tokens = [w for s in error_tokens for w in s]
    
    if type(correct_tokens[0]) == str:
        corrected = 0
        errors = 0
        for c,p,e in zip(correct_tokens,predicted_tokens,error_tokens):
            if c!=e:
                
                alignment1 = align(e,c)
                alignment2 = align(p,c)
                e = sum([True if (t[0]!=t[1]) or (None in t) else False for t in alignment1])
                errors+=e
                corrected += abs(sum([True if ((t[0]!=t[1]) or (None in t)) else False for t in alignment2])-e)
#                 print(e,corrected)

        return round(corrected/errors,3),corrected,errors