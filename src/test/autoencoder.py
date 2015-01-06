'''
Created on Oct 10, 2014

@author: sadegh
'''
import theano, sys
import numpy
import time
import theano.tensor as T
from numpy import dtype


class AutoEncoder(object):

    def __init__(self, numpy_rng, input=None, n_visible=10, n_hidden=5, W=None, bhid=None, bvis=None):
        print "in __init__ function..."
        """
    
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights
    
    
        :type input: theano.tensor.TensorType
        :paran input: a symbolic description of the input or None for standalone dA
    
        :type n_visible: int
        :param n_visible: number of visible units
    
        :type n_hidden: int
        :param n_hidden:  number of hidden units
    
        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
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
        self.n_visible = n_visible
        self.n_hidden = n_hidden
    
    
        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and 4*sqrt(6./(n_hidden+n_visible))
            # the output of uniform if converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)), dtype=theano.config.floatX)  # @UndefinedVariable
            W = theano.shared(value=initial_W, name='W')
    
        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible,
                                        dtype=theano.config.floatX), name='bvis')  # @UndefinedVariable
    
        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden,
                                              dtype=theano.config.floatX), name='bhid')  # @UndefinedVariable
        
        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several examples,
            # each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
    
        self.params = [self.W, self.b, self.b_prime]
        print "W:"
        print self.W.get_value()
        print "b:"
        print self.b.get_value()
        print "b_prime:"
        print self.b_prime.get_value()
    
    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
#         print T.dot(input, self.W).eval()
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)
    
    def get_reconstructed_input(self, hidden):
        """ Computes the reconstructed input given the values of the hidden layer """
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
    
    def get_cost_updates(self, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step """
    
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        
        # note : we sum over the size of a datapoint; if we are using minibatches,
        #        L will  be a vector, with one entry per example in minibatch
        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
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
        print updates[0:4]
        return (self.x, z, L, cost, updates)

def load_data(dataset):
    print "load data"


def test_AE(learning_rate=0.1, training_epochs=10, batch_size=2):
    ######################
    # BUILDING THE MODEL #
    ######################
    train_set_x = numpy.array(numpy.random.randint(2, size = 30), dtype = theano.config.floatX).reshape((6, 5))  # @UndefinedVariable
    train_set_x = theano.shared(train_set_x)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
#     n_train_batches = train_set_x.shape[0] / batch_size
    print n_train_batches
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
#     index = 0
    x = T.matrix('x')  # the data is presented as rasterized images
    
    autoencoder = AutoEncoder(numpy_rng=numpy.random.RandomState(1234), n_visible=5, n_hidden=3, input = x)
    inputLayer, outputLayer, L, cost, updates = autoencoder.get_cost_updates(learning_rate)
    train = theano.function(inputs = [index], outputs = [inputLayer, outputLayer, L, cost], updates=updates, 
                            givens = {x: train_set_x[index * batch_size: (index + 1) * batch_size, :]})
    
    start_time = time.clock()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            train_set_x.set_value((numpy.array(numpy.random.randint(2, size = 30), dtype = theano.config.floatX).reshape((6, 5))))  # @UndefinedVariable
#             train_set_x = numpy.array(numpy.random.randint(2, size = 30), dtype = theano.config.floatX).reshape((6, 5))  # @UndefinedVariable
#             train_set_x = theano.shared(train_set_x)
            t = train(batch_index)
            c.append(t[2])
            print "\n x: \n"
            print t[0]
            print "\n z: \n"
            print t[1]
            print "\n L: \n"
            print t[2]
        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)
    
    end_time = time.clock()
    training_time = (end_time - start_time)
    
    print ('Training took %f minutes' % (training_time / 60.))

if __name__ == '__main__':
    test_AE()

