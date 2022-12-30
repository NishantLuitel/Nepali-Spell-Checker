from knlm import KneserNey
import regex as re


# with open('data/saved_km_lm2.pickle','rb') as outputfile:
#     kn_lm = pickle.load(outputfile)


mode = 'build'
if mode == 'build':
    # build model from corpus text. order = 3, word size = 4 byte
    mdl = KneserNey(3, 4)
    for line in open('data/compiled.txt', encoding='utf-8'):
        line = re.findall(r'[\u0900-\u097F]+')
        mdl.train(line.strip().split())
    mdl.optimize()
    mdl.save('language_knlm.model')
else:
    # load model from binary file
    mdl = KneserNey.load('language_knlm.model')
    print('Loaded')
print('Order: %d, Vocab Size: %d, Vocab Width: %d' % (mdl.order, mdl.vocabs, mdl._wsize))

# evaluate sentence score
# print(mdl.evaluateSent('I love kiwi .'.split()))
# print(mdl.evaluateSent('ego kiwi amo .'.split()))

# evaluate scores for each word
# print(mdl.evaluateEachWord('I love kiwi .'.split()))
# print(mdl.evaluateEachWord('ego kiwi amo .'.split()))