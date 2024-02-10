# Nepali-
Nepali spelling correction

# Model Training




# Error Generation



# Experiments

The experiments we performed can either be redone by running the notebooks on the experiments directory. 
There are five notebooks and each notebook is named under the type of language model used for training. 
Hence each notebook runs at least two experiments that include both types of error models. Moreover, Trans and Trans-word 
notebooks contain experiments we performed for the ablation study.
The experiments can also be performed using command lines.

Activate your virtual environment and install requirements

```{python}
pip install -r requirements.txt
```

For example, to run an experiment where Trans-word LM is used with a CD error model use:
```{python}
python experiments/exp.py -l Trans-word -e cd
```
To further give weighting to LM you can set lambda value using:
```{python}
python experiments/exp.py -l Trans-word -e cd --lambda 1.2
```

To perform an ablation study, just set the ablation parameter and choose one of either 'Trans' or 'Trans-word' LM for 
experimenting with either word-based or sentence-based techniques. 
The following code performs an ablation study using the BM model for 323 sentences.
```{python}
python experiments/exp.py --ablation -l Trans -e bm --num 323
```

# Command Line Options:

Options:
  -h, --help            show this help message and exit
  --ablation, --Ablation
                        perform ablation study
  -l LM, --LM=LM        set the variant of the language model to evaluate with
                        options: knlm, NepaliBert,dBerta, Trans and Trans-word
  -e EM, --error_model=EM
                        select an error model to train with option: 'cd' for
                        constant distributive and 'bm' for Brill and Moore
                        version
  -f EVAL_FILE, --file=EVAL_FILE
                        set the filename for evaluation
  --num=NUM, --number=NUM
                        set the number of sentences to perform evaluation over
                        default: 400 or max
  --lda=P_LAMDA, --lambda=P_LAMDA
                        sets the weighting factor for the language model(prior)
                        default:1.0
  --trie, --Trie        set this to make use of an efficient Trie data
                        structure during candidate selection

Note: Models Required for the project are uploaded here.
