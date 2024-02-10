


if __name__ == '__main__':
    import os
    import sys
    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    sys.path.append(parent_dir)
    
    
    from optparse import OptionParser  
    
    parser = OptionParser()
    parser.add_option("-l", "--LM", dest="lm",
                      help="write report to FILE")
    parser.add_option("-e", "--error_model",
                       dest="em", default="cd",
                      help="select an error model to train with\
                            opions: 'cd' for constant distributive and 'bm' for Brill and Moore version")
    parser.add_option("-f", "--file",
                       dest="eval_file", default="data/eval_data2.pic",
                      help="Hello")
    parser.add_option( "--lda","--lamda",
                       dest="p_lamda", type = "float", default=1.0,
                      help="Hello2")
    parser.add_option("--trie", "--Trie",
                      action="store_true", dest="trie", default=False,
                      help="don't print status messages to stdout")

    (options, args) = parser.parse_args()
    
    if options.lm == "transformer":
        
        from transformer import model
        print(model)
        
    
    
    
    
    