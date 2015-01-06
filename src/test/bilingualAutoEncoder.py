'''
Created on Oct 17, 2014

@author: sadegh
'''
import theano, sys
import nltk, string, codecs
from nltk.stem.wordnet import WordNetLemmatizer
import numpy
import scipy.sparse as sp
import time
import theano.tensor as T
from numpy import dtype
from theano import sparse
import collections
import operator

class AutoEncoder(object):

    def __init__(self, numpy_rng, input=None, translation=None, n_first=10, n_hidden=3, n_last=7, W1=None, W2 = None, b1=None, b2_en=None, b2_de=None):
        print "in __init__ function..."
        """
        
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights
    
    
        :type input: theano.tensor.TensorType
        :paran input: a symbolic description of the input or None for standalone dA
    
        :type n_first: int
        :param n_first: number of units in the first layer
        
        :type n_last: int
        :param n_last: number of units in the last layer
    
        :type n_hidden: int
        :param n_hidden:  number of hidden units
    
        :type W1: theano.tensor.TensorType
        :param W1: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None
                  
        :type W2: theano.tensor.TensorType
        :param W2: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None
    
        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
    
        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
    
    
        """
        self.n_first = n_first
        self.n_hidden = n_hidden
    
    
        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W1:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and 4*sqrt(6./(n_hidden+n_visible))
            # the output of uniform if converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W1 = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_first)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_first)),
                      size=(n_hidden, n_first)), dtype=theano.config.floatX)  # @UndefinedVariable
            W1 = theano.shared(value=initial_W1, name='W1')
            
        if not W2:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and 4*sqrt(6./(n_hidden+n_visible))
            # the output of uniform if converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W2 = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_last)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_last)),
                      size=(n_hidden, n_last)), dtype=theano.config.floatX)  # @UndefinedVariable
            W2 = theano.shared(value=initial_W2, name='W2')
    
        if not b1:
            b1 = theano.shared(value=numpy.zeros(n_hidden,
                                        dtype=theano.config.floatX), name='b1')  # @UndefinedVariable
    
        if not b2_en:
            b2_en = theano.shared(value=numpy.zeros(len(wordToIndEn),
                                              dtype=theano.config.floatX), name='b2_en')  # @UndefinedVariable
        if not b2_de:
            b2_de = theano.shared(value=numpy.zeros(len(wordToIndDe),
                                              dtype=theano.config.floatX), name='b2_de')  # @UndefinedVariable
        
        self.W1 = W1
        self.W2 = W2
        # b corresponds to the bias of the hidden
        self.b1 = b1
        # b_prime corresponds to the bias of the visible
        self.b2_en = b2_en
        self.b2_de = b2_de
        
        # NO tied weights, therefore W_prime is NOT W transpose
        
        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several examples,
            # each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
            
        if translation == None:
            self.actualTrans = T.dmatrix(name='trnl')
        else:
            self.actualTrans = translation
    
        self.params = [self.W1, self.W2, self.b1, self.b2_en, self.b2_de]
        print "W1:"
        print self.W1.get_value()
        print "W2:"
        print self.W2.get_value()
        print "b1:"
        print self.b1.get_value()
        print "b2_en:"
        print self.b2_en.get_value()
        print "b2_de:"
        print self.b2_de.get_value()
    
    def get_hidden_values(self, input, inputLang):
        """ Computes the values of the hidden layer """
#         print T.dot(input, self.W).eval()
        if inputLang == "en":
            return T.nnet.sigmoid(T.dot(input, self.W1.T) + self.b1)
        if inputLang == "de":
            return T.nnet.sigmoid(T.dot(input, self.W2.T) + self.b1)
    
    def get_reconstructed_input(self, hidden, outputLang):
        """ Computes the reconstructed input given the values of the hidden layer """
        if outputLang == "en":
            return T.nnet.sigmoid(T.dot(hidden, self.W1) + self.b2_en)
        if outputLang == "de":
            return T.nnet.sigmoid(T.dot(hidden, self.W2) + self.b2_de)
    
    def get_cost_updates(self, learning_rate):
        """ This function computes the cost and the updates for one training
        step """
        # note : we sum over the size of a datapoint; if we are using minibatches,
        #        L will  be a vector, with one entry per example in minibatch
        # MonoLingual Mode
        y = self.get_hidden_values(self.x, "en")
        z = self.get_reconstructed_input(y, "de")
        L = -T.sum(self.actualTrans * T.log(z) + (1 - self.actualTrans) * T.log(1 - z), axis=1)
        
        y = self.get_hidden_values(self.actualTrans, "de")
        z = self.get_reconstructed_input(y, "en")
        L += -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        
        
        # BilingualMode
        y = self.get_hidden_values(self.x, "en")
        z = self.get_reconstructed_input(y, "en")
        L += -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        
        y = self.get_hidden_values(self.actualTrans, "de")
        z = self.get_reconstructed_input(y, "de")
        L += -T.sum(self.actualTrans * T.log(z) + (1 - self.actualTrans) * T.log(1 - z), axis=1)
        
        # correlation
        L += -T.sum(T.nnet.sigmoid(T.dot(self.x, self.W1.T) + self.b1) * 
                    T.nnet.sigmoid(T.dot(self.actualTrans, self.W2.T) + self.b1), axis=1)
        
#         actualTrans = theano.shared(numpy.array(numpy.random.randint(2, size = 2 * 7), 
#                                   dtype = theano.config.floatX).reshape((2, 7)))  # @UndefinedVariable
        
        # note : L is now a vector, where each element is the cross-entropy cost
        #        of the reconstruction of the corresponding example of the
        #        minibatch. We need to compute the average of all these to get
        #        the cost of the minibatch
        cost = T.mean(L)
#         print "cost:"
#         print cost.eval()
    
        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        print updates[0:5]
        return (self.x, z, cost, L, self.actualTrans, updates, self.W1, self.W2, self.b1, self.b2_en, self.b2_de)

def lemmatizeEnInd():
    
    lmtzr = WordNetLemmatizer()
    print lmtzr.lemmatize('cars')
    print len(wordToIndEn)
    s = set()
    for w in wordToIndEn:
        s.add(lmtzr.lemmatize(w))
    print len(s)

def removeLowFreq():
#     pathRead = "/home/sadegh/workspace/DocRep/data/de-en-europarl/europarl-v7-clean.de-en.de"
#     pathWrite = "/home/sadegh/workspace/DocRep/data/de-en-europarl/europarl-v7-clean-highFreq.de-en.de"
    pathRead = "/home/sadegh/workspace/DocRep/data/de-en-europarl/en-highFreq-split/EN-000"
    pathWrite = "/home/sadegh/workspace/DocRep/data/de-en-europarl/EN-000-highFreq"
    
    with codecs.open(pathRead, "r", encoding="utf8") as f:
        content = f.readlines()
    content = [x.strip('\n') for x in content]
    i = 0
    print "Read Complete"
    with codecs.open(pathWrite, "w+", encoding="utf8") as fWrite:
        for line in content:
            tokens = line.split()
            post = ""
            for w in tokens:
                if enWordToCount[w] >= 3:
                    post += w + " "
            content[i] = post.strip()
            fWrite.write("%s\n" % content[i])
            print i
            i += 1
    print "Write Complete"
    
    pathRead = "/home/sadegh/workspace/DocRep/data/de-en-europarl/de-highFreq-split/DE-000"
    pathWrite = "/home/sadegh/workspace/DocRep/data/de-en-europarl/DE-000-highFreq"
    
    with codecs.open(pathRead, "r", encoding="utf8") as f:
        content = f.readlines()
    content = [x.strip('\n') for x in content]
    i = 0
    print "Read Complete"
    with codecs.open(pathWrite, "w+", encoding="utf8") as fWrite:
        for line in content:
            tokens = line.split()
            post = ""
            for w in tokens:
                if deWordToCount[w] >= 3:
                    post += w + " "
            content[i] = post.strip()
            fWrite.write("%s\n" % content[i])
            print i
            i += 1
    print "Write Complete"


enWordToCount = {}
deWordToCount = {}
def writeWordCount():
    global enWordToCount
    global deWordToCount
    pathReadEn = "/home/sadegh/workspace/DocRep/data/de-en-europarl/EN-000-highFreq"
    with codecs.open(pathReadEn, "r", encoding="utf8") as f:
        for line in f:
            for word in line.split():
                if enWordToCount.has_key(word):
                    enWordToCount[word] += 1
                else:
                    enWordToCount[word] = 1
    print "English Word Counts constructed..."
    
    pathWrite = "/home/sadegh/workspace/DocRep/data/de-en-europarl/wcIndex-small.en"
    with codecs.open(pathWrite, "w+", encoding="utf8") as fWrite:
        for t in sorted(enWordToCount.items(), key=operator.itemgetter(1)):
            fWrite.write("%s, %s\n" % (t[0], str(t[1])))
     
    pathReadDe = "/home/sadegh/workspace/DocRep/data/de-en-europarl/DE-000-highFreq"
    with codecs.open(pathReadDe, "r", encoding="utf8") as f:
        for line in f:
            for word in line.split():
                if deWordToCount.has_key(word):
                    deWordToCount[word] += 1
                else:
                    deWordToCount[word] = 1
    print "German Word Counts constructed..."
    pathWrite = "/home/sadegh/workspace/DocRep/data/de-en-europarl/wcIndex-small.de"
    with codecs.open(pathWrite, "w+", encoding="utf8") as fWrite:
        for t in sorted(deWordToCount.items(), key=operator.itemgetter(1)):
            fWrite.write("%s, %s\n" % (t[0], str(t[1])))
#             fWrite.write("%s\n" % (t[0]))
    

def getChunkBoW(chkInd, chunkSize):
    ts_x = numpy.zeros((chunkSize, len(wordToIndEn)), dtype=numpy.int8)
    ts_y = numpy.zeros((chunkSize, len(wordToIndDe)), dtype=numpy.int8)
    j = 0
    for i in range(chkInd * chunkSize, (chkInd + 1) * chunkSize):
        for w in enSents[i].split():
            ts_x[j,wordToIndEn[w]] = 1
        for w in deSents[i].split():
            ts_y[j,wordToIndDe[w]] = 1
        j += 1
    return (ts_x, ts_y)


def constructBinaryBoW(lang, sentence):
    if lang == "en":
        bowEn = numpy.zeros(len(wordToIndEn), dtype=numpy.int8)
        for w in sentence.split():
            bowEn[wordToIndEn[w]] = 1
            return bowEn
    if lang == "de":
        bowDe = numpy.zeros(len(wordToIndDe), dtype=numpy.int8)
        for w in sentence.split():
            bowDe[wordToIndDe[w]] = 1
            return bowDe
    

wordToIndEn = {}
wordToIndDe = {}
def loadIndex():
    global wordToIndEn
    global wordToIndDe
    
    pathReadEn = "/home/sadegh/workspace/DocRep/data/de-en-europarl/index-EN-highFreq-small"
    with codecs.open(pathReadEn, "r", encoding="utf8") as f:
        enIndex = f.readlines()
    enIndex = [x.strip('\n') for x in enIndex]
    i = 0
    for w in enIndex:
        wordToIndEn[w] = i
        i += 1
    
    pathReadDe = "/home/sadegh/workspace/DocRep/data/de-en-europarl/index-DE-highFreq-small"
    with codecs.open(pathReadDe, "r", encoding="utf8") as f:
        deIndex = f.readlines()
    deIndex = [x.strip('\n') for x in deIndex]
    i = 0
    for w in deIndex:
        wordToIndDe[w] = i
        i += 1
    
    print "EN and DE index loaded successfully..."

enSents = []
deSents = []
def loadCorpora(part):
    part = str(part)
    print "part " + part + " loading.."
    if len(part) == 1:
        part = "00" + part
    elif len(part) == 2:
        part = "0" + part
    global enSents
    global deSents
#     pathReadEn = "/home/sadegh/workspace/DocRep/data/de-en-europarl/en-highFreq-split/EN-" + part
    pathReadEn = "/home/sadegh/workspace/DocRep/data/de-en-europarl/EN-000-highFreq"
    with codecs.open(pathReadEn, "r", encoding="utf8") as f:
        enSents = f.readlines()
    enSents = [x.strip('\n') for x in enSents]
    
#     pathReadDe = "/home/sadegh/workspace/DocRep/data/de-en-europarl/de-highFreq-split/DE-" + part
    pathReadDe = "/home/sadegh/workspace/DocRep/data/de-en-europarl/DE-000-highFreq"
    with codecs.open(pathReadDe, "r", encoding="utf8") as f:
        deSents = f.readlines()
    deSents = [x.strip('\n') for x in deSents]
    
    print part + "\n"
    print len(enSents)
    print str(len(deSents)) + "\n"

# tokenize, remove punc and then lowercase
def preprocess():
    pathRead = "/home/sadegh/workspace/DocRep/data/de-en-europarl/europarl-v7.de-en.de"
    pathWrite = "/home/sadegh/workspace/DocRep/data/de-en-europarl/europarl-v7-clean.de-en.de"
    
    with codecs.open(pathRead, "r", encoding="utf8") as f:
        content = f.readlines()
    content = [x.strip('\n') for x in content]
    i = 0
    print "Read Complete"
    with codecs.open(pathWrite, "w+", encoding="utf8") as fWrite:
        for line in content:
            tokens = nltk.word_tokenize(line.strip())
            post = ""
            for w in tokens:
                if w not in string.punctuation:
                    post += w.lower() + " "
            content[i] = post.strip()
            fWrite.write("%s\n" % content[i])
            print i
            i += 1
    print "Write Complete"

def createIndex():
#     pathRead = "/home/sadegh/workspace/DocRep/data/de-en-europarl/europarl-v7-clean-highFreq.de-en.en"
#     pathWrite = "/home/sadegh/workspace/DocRep/data/de-en-europarl/index-EN-highFreq"
    pathRead = "/home/sadegh/workspace/DocRep/data/de-en-europarl/EN-000-highFreq"
    pathWrite = "/home/sadegh/workspace/DocRep/data/de-en-europarl/index-EN-highFreq-small"
#     pathRead = "/home/sadegh/workspace/DocRep/data/de-en-europarl/en-split/EN-000"
    s = sorted(set(w for w in codecs.open(pathRead, "r", encoding="utf8").read().split()))
    
    with codecs.open(pathWrite, "w+", encoding="utf8") as fWrite:
        for w in s:
            fWrite.write("%s\n" % w)
    print len(s)
    print "English Finished"
    
#     pathRead = "/home/sadegh/workspace/DocRep/data/de-en-europarl/europarl-v7-clean-highFreq.de-en.de"
#     pathWrite = "/home/sadegh/workspace/DocRep/data/de-en-europarl/index-DE-highFreq"
    pathRead = "/home/sadegh/workspace/DocRep/data/de-en-europarl/DE-000-highFreq"
    pathWrite = "/home/sadegh/workspace/DocRep/data/de-en-europarl/index-DE-highFreq-small"
    s = sorted(set(w for w in codecs.open(pathRead, "r", encoding="utf8").read().split()))
    with codecs.open(pathWrite, "w+", encoding="utf8") as fWrite:
        for w in s:
            fWrite.write("%s\n" % w)
    print len(s)
    print "German Finished"
    

def test_AE(learning_rate=0.1, training_epochs=10, batch_size=5):
    theano.config.exception_verbosity = 'high'
    ######################
    # BUILDING THE MODEL #
    ######################
#     train_set_x = numpy.array(numpy.random.randint(2, size = 20), dtype = theano.config.floatX).reshape((4, 5))  # @UndefinedVariable
    train_set_x = numpy.array(numpy.random.randint(2, size = 20), dtype = numpy.int8).reshape((4, 5))
    train_set_x = theano.shared(train_set_x)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    print n_train_batches
    
    train_set_trans = theano.shared(numpy.array(numpy.random.randint(2, size = 28), dtype = numpy.int8).reshape((4, 7)))
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.bmatrix('x', )  # the data is presented as rasterized images
    trans = T.bmatrix('trans')
    ### ---> return (self.x, z, cost, L, self.actualTrans, updates, self.W1, self.W2, self.b1, self.b2_en, self.b2_de)
    autoencoder = AutoEncoder(numpy_rng=numpy.random.RandomState(1234), n_first=len(wordToIndEn),
                               n_hidden=20, n_last = len(wordToIndDe), input = x, translation = trans)
    inputLayer, outputLayer, cost, L, actualTrans, updates, W_1, W_2, b_1, b_2_en, b_2_de = autoencoder.get_cost_updates(learning_rate)
    train = theano.function(inputs = [index], outputs = [inputLayer, outputLayer, cost, L, actualTrans, W_1, W_2, b_1, b_2_en, b_2_de],
                             updates=updates, givens = {x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                      trans: train_set_trans[index * batch_size: (index + 1) * batch_size]})
    
    start_time = time.clock()

    ############
    # TRAINING #
    ############
    chunkSize = 1000
    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        for p in range(1):
            loadCorpora(p)
            
            for chkInd in range(len(enSents)/chunkSize):
                ts_x, ts_trans = getChunkBoW(chkInd, chunkSize)
                train_set_x.set_value(ts_x)
                train_set_trans.set_value(ts_trans)
                n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
                c = []
                for batch_index in xrange(n_train_batches):
                    t = train(batch_index)
                    c.append(t[2])
#                     print "\n x: \n"
#                     print t[0]
#                     print "\n z: \n"
#                     print t[1]
#                     print "\n actualTrans: \n"
#                     print t[4]
#                     print "\n L: \n"
#                     print t[3]
                    print "batch: " + str(batch_index) + " done... (batch size: " + str(batch_size) + ")"
                print "===>>> chunk: " + str(chkInd) + " from part: " + str(p) + " done... (total chunks in this part: " + str(len(enSents)/chunkSize) + ")"
            print "\n\n======================================================================\n\n"
            print "epoch: " + str(epoch) + ", part: " + str(p) + " done.."
        print "\n\n\n**********************************************************************************************\n\n\n"
        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)
    
    ### ---> outputs = [inputLayer, outputLayer, cost, L, actualTrans, W_1, W_2, b_1, b_2_en, b_2_de]
    end_time = time.clock()
    training_time = (end_time - start_time)
    
    numpy.save("W1", t[5])
    numpy.save("W2", t[6])
    
    print ('Training took %f minutes' % (training_time / 60.))


if __name__ == '__main__':
#     pathRead = "/home/sadegh/workspace/DocRep/data/de-en-europarl/europarl-v7-clean.de-en.en"
#     print len(set(w for w in open(pathRead).read().split()))
    
#     for i in range(193):
#         loadCorpora(str(i))
#     createIndex()
    loadIndex()
#     writeWordCount()
#     removeLowFreq()
#     lemmatizeEnInd()
    test_AE()
#     sys.exit(0)

