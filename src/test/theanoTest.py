'''
Created on Oct 9, 2014

@author: sadegh
'''
import nltk
import theano
import numpy
import codecs, operator

indToWordEn = {}
indToWordDe = {}

def loadInd():
    global indToWordEn
    global indToWordDe
    pathReadEn = "/home/sadegh/workspace/DocRep/data/de-en-europarl/index-EN-highFreq-small"
    with codecs.open(pathReadEn, "r", encoding="utf8") as f:
        enIndex = f.readlines()
    enIndex = [x.strip('\n') for x in enIndex]
    i = 0
    for w in enIndex:
        indToWordEn[i] = w
        i += 1
    
    pathReadDe = "/home/sadegh/workspace/DocRep/data/de-en-europarl/index-DE-highFreq-small"
    with codecs.open(pathReadDe, "r", encoding="utf8") as f:
        deIndex = f.readlines()
    deIndex = [x.strip('\n') for x in deIndex]
    i = 0
    for w in deIndex:
        indToWordDe[i] = w
        i += 1

if __name__ == '__main__':
    print theano.config.floatX  # @UndefinedVariable
    loadInd()
    w1 = numpy.load("W1.npy")
    w2 = numpy.load("W2.npy")
    january = w1[:,932]
    wordToDist = {}
    for i in range(w1.shape[1]):
        wordToDist[indToWordEn[i]] = numpy.linalg.norm(january - w1[:,i])
    pathWrite = "/home/sadegh/workspace/DocRep/data/de-en-europarl/oilNeighboursEn"
    i = 0
    for t in sorted(wordToDist.items(), key=operator.itemgetter(1)):
        i += 1
        print t[0] + "\t" + str(t[1])
        if i == 15:
            break
    print "\n"
    wordToDist = {}
    for i in range(w2.shape[1]):
        wordToDist[indToWordDe[i]] = numpy.linalg.norm(january - w2[:,i])
#     pathWrite = "/home/sadegh/workspace/DocRep/data/de-en-europarl/oilNeighboursDe"
#     with codecs.open(pathWrite, "w+", encoding="utf8") as fWrite:
#         for t in sorted(wordToDist.items(), key=operator.itemgetter(1)):
#             fWrite.write("%s, %s\n" % (t[0], str(t[1])))
    i = 0
    for t in sorted(wordToDist.items(), key=operator.itemgetter(1)):
        i += 1
        print t[0] + "\t" + str(t[1])
        if i == 15:
            break
        
#     sentence = """At eight o'clock on Thursday morning."""
#     tokens = nltk.word_tokenize(sentence)
#     print tokens
#     nltk.download()
