
import regex as re
from collections import Counter
import pickle

# def words(text): return re.findall(r'\w+', text.lower())


def words(text): return re.findall(r'[\u0900-\u097F]+', text.lower())


def words_bigram(text): return [tuple(x.split()) for x in re.findall(
    r'\b[\u0900-\u097F]+\s[\u0900-\u097F]+', text.lower(), overlapped=True)]

def words_trigram(text): return re.findall(
    r'\b[\u0900-\u097F]+\s[\u0900-\u097F]+\s[\u0900-\u097F]+', text.lower(),
    overlapped=True)


WORDS = Counter(words(open('data/compiled.txt', encoding="utf-8").read()))
with open('data/saved_words.pickle','wb') as outputfile:
    pickle.dump(WORDS,outputfile)


WORDS_bigram = Counter(words_bigram(
    open('data/compiled.txt', encoding="utf-8").read()))


# def save_counter_bigram()
#     i = 0
#     start_l = ['<s>']*(n-1)
#     start = '<s>\u0020'*(n-1)
#     with open('data/compiled.txt', encoding='utf-8') as text:
#         text = re.sub(r'\s+[\u0964]\s+', r'\u0020\u0964\u0020'+start, text)
#         #text = re.findall(r'[\u0900-\u097F]+|\u003c\u002f\u0073\u003e|\u003c\u0073\u003e',text.lower().strip())
#         text = start_l + text
#         counter = Counter([tuple(x.split()) for x in re.findall(r'\s+[\u0900-\u097F]+\s+[\u0900-\u097F]+|\u003c\u0073\u003e\s+[\u0900-\u097F]+|[\u0900-\u097F]+\s+\u003c\u0073\u003e', line, overlapped=True)])
#     return counter

# WORDS = Counter(words_bigram(text))
# with open('data/saved_words_counter2.pickle','wb') as outputfile:
#     pickle.dump(WORDS,outputfile)


with open('data/saved_bi_words.pickle','wb') as outputfile:
    pickle.dump(WORDS_bigram,outputfile)
    
WORDS_trigram = Counter(words_trigram(
    open('data/compiled.txt', encoding="utf-8").read()))

with open('data/saved_tri_words.pickle','wb') as outputfile:
    pickle.dump(WORDS_trigram,outputfile)

# List of all Nepali characters
char_vocab = []
for _ in range(2304, 2432):
    char_vocab += [chr(_)]


def probability(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N


def probability_bigram(bi_word, N=sum(WORDS_bigram.values())):
    "Probability of `two words` given as a tuple."
    return (WORDS_bigram[bi_word]+1) / N

def probability_trigram(tri_word, N=sum(WORDS_trigram.values())):
    "Probability of `two words` given as a tuple."
    return (WORDS_trigram[tri_word]+1) / N


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
    return (known([word]) or known(edits1(word)) or
            known(edits2(word)) or [word])


def candidates_all(word):
    "Generate possible spelling corrections for word."
    return (set.union(known([word]), known(edits1(word)),
                      known(edits2(word)), known(word)))


def correction(word):
    "Most probable spelling correction for word."
    return max(candidates_ordered(word), key=probability)


def correction1(word):
    "Most probable spelling correction for word."
    return max(candidates_all(word), key=probability)
