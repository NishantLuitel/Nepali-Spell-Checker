# Nepali-
Nepali spelling correction

## Language Models

1. **KnLM-2**: We have also trained a probabilistic bi-gram language model with Kneser-Ney smoothing which allows backing off to unigrams when bigram isn’t available. Inference using n-gram language models is fast because they involve dictionary lookup. However, the memory requirement increases drastically with increasing 'n' so it is not efficient to capture longer contexts. We use candidate sentence ranking with this model.

2. **NepaliBert**: NepaliBert is a pre-trained LM trained in Masked Language modeling fashion using the BERT architecture. We use candidate sentence ranking with this model.

3. **deBerta**: deBerta is the Bert-based Nepali LM trained by following the approach used in deBerta LM. Basically, instead of using the sum of positional encoding with the embedding of tokens, the deBert model uses a separate embedding to positional encoding as well and uses the disentangled attention calculated from both relative position and content. Like NepaliBert, we again use candidate sentence ranking.

4. **Trans**: We have trained an autoregressive LM with 4 transformer encoders and the same number of decoders on the word-based vocabulary of around 350,000 using the Oscar corpus. An embedding size of 300 was used with four attention heads in the transformer. The dropout value was 0.05. It was trained with cross-entropy loss. The Trans model also uses candidate sentence correction.

5. **Trans-word**: This is the same language model as described in 4. However, it uses candidate word correction rather than candidate sentences.

## Error Models
1. **Constant distributive (CD)**: In this error model, we set an 'α' parameter which represents the probability that an observed word is correct, i.e., P(w|w). Then, we uniformly distribute '1−α' over all the candidates of 'x' except 'w'. Mathematically, the absolute value of x is defined as:
    ```
    P(x|w) =
    {
        α                         if x = w
        (1−α) / |C(x)|−1         if x ∈ C(x)
        0                         if x /∈ C(x)
    ```
    We choose the value of α to be 0.65.

2. **Brill and Moore (BM)**: BM error model was trained as described in section III-B with two corpus of data. Training performed in 'Oscar' dataset was used for evaluation.



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

# Further Command Line Options:
Options:
- `-h, --help`: Show this help message and exit.
- `--ablation, --Ablation`: Perform ablation study.
- `-l LM, --LM=LM`: Set the variant of language model to evaluate with. Options: `knlm`, `NepaliBert`, `dBerta`, `Trans`, and `Trans-word`.
- `-e EM, --error_model=EM`: Select an error model to train with. Options: `'cd'` for constant distributive and `'bm'` for Brill and Moore version.
- `-f EVAL_FILE, --file=EVAL_FILE`: Set the filename for evaluation.
- `--num=NUM, --number=NUM`: Set the number of sentences to perform evaluation over. Default: 400 or max.
- `--lda=P_LAMDA, --lambda=P_LAMDA`: Sets the weighting factor for language model (prior). Default: 1.0.
- `--trie, --Trie`: Set this to make use of an efficient Trie data structure during candidate selection.


Note: Models Required for the project are uploaded here.
