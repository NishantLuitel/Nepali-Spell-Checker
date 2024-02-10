


if __name__ == '__main__':
    import os
    import sys
    import torch
    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    sys.path.append(parent_dir)
    from eval import gather_dataset,WER,word_accuracy,char_accuracy
    from optparse import OptionParser  
    from corrector import extract_choices,autocorrect_word,words
    
    
    parser = OptionParser()
    parser.add_option("--ablation", "--Ablation",
                      action="store_true", dest="ablation", default=False,
                      help="perform ablation study")
    parser.add_option("-l", "--LM", dest="lm", default='Trans',
                      help="set the variant of language model to evaluate with\
                      options: knlm,NepaliBert,dBerta,Trans and Trans-word")
    parser.add_option("-e", "--error_model",
                       dest="em", default="cd",
                      help="select an error model to train with\
                            opions: 'cd' for constant distributive and 'bm' for Brill and Moore version")
    parser.add_option("-f", "--file",
                       dest="eval_file", default="eval_data2.pic",
                      help="set the filename for evaluation")
    parser.add_option( "--num","--number",
                       dest="num", type = "int", default=400,
                      help="set the number of sentences to perform evaluation over\
                      default: 400 or max")
    parser.add_option( "--lda","--lambda",
                       dest="p_lamda", type = "float", default=1.0,
                      help="sets the weighting factor for language model(prior)\
                      default:1.0")
    parser.add_option("--trie", "--Trie",
                      action="store_true", dest="trie", default=False,
                      help="set this to make use of a efficient Trie data structure during candidate selection")
    
    (options, args) = parser.parse_args()
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    tokenizer=None
    if options.lm == 'NepaliBERT':

        from transformers import BertTokenizer
        vocab_file_dir = './NepaliBERT/' 
        tokenizer = BertTokenizer.from_pretrained(vocab_file_dir,
                                                strip_accents=False,
                                                   clean_text=False )
        from transformers import BertForMaskedLM
        model = BertForMaskedLM.from_pretrained('./NepaliBERT').to(device)
        model_type = 'bert'
        
    elif options.lm == 'deBerta':
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        tokenizer = AutoTokenizer.from_pretrained('Sakonii/deberta-base-nepali')
#         print(type(tokenizer))
        model = AutoModelForMaskedLM.from_pretrained('Sakonii/deberta-base-nepali')
        model_type = 'bert'
        
    elif options.lm == 'knlm2':
        import pickle
        with open(os.path.join(current_dir, '..','models','saved_model_knlm2'),'rb') as inputfile:
            model = pickle.load(inputfile) 
        model_type = 'knlm'
            
    elif options.lm == 'Trans':
        from transformer import model  
        model_type= 'transformer'
            
    elif options.lm == 'Trans-word':
        from transformer import model 
        model_type = 'transformer'
        extract_choices = autocorrect_word
        
    # Likelihood fix
    likelihood = 'bm' if options.em == 'bm' else 'default'
        
   # Form the dataset        
    dataset_file = os.path.join(current_dir, '..','data',options.eval_file)
    dataset = gather_dataset(dataset_file)[:options.num]
    correct_tokens = [words(t[0]) for t in dataset]
    error_tokens = [words(t[1]) for t in dataset]
    error_sentences = [t[1] for t in dataset] 
    
    
    predicted_tokens = []    
    for i,s in enumerate(error_sentences):        
       
        
        c = extract_choices(s,model=model,p_lambda =options.p_lamda ,trie = True,likelihood = likelihood,model_type=model_type,tokenizer=tokenizer,ab=options.ablation)
#         print(c)
        c = [t[0] for t in c if len(t)>=1]
        predicted_tokens.append(c)
        if i%2 == 0:
            
            a = len(predicted_tokens)
            # print current stat
            print(i,') ','WER observed: ',WER(correct_tokens[:a],error_tokens[:a]),
                  'WER corrected: ',WER(correct_tokens[:a],predicted_tokens[:a]),
                  'word acc.: ', word_accuracy(correct_tokens[:a],predicted_tokens[:a],error_tokens[:a]),
                  'char acc.: ',char_accuracy(correct_tokens[:a],predicted_tokens[:a],error_tokens[:a]))
        if i==options.num-1:
            
            l = len(predicted_tokens)
            print('WER observed: ',WER(correct_tokens[:l],error_tokens[:l]),
                  'WER corrected: ',WER(correct_tokens[:l],predicted_tokens[:l] ),
                  'word acc.: ', word_accuracy(correct_tokens[:l],predicted_tokens[:l],error_tokens[:l]),
                  'char acc.: ',char_accuracy(correct_tokens[:l],predicted_tokens[:l],error_tokens[:l]))

    
    
    
    
    