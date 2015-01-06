import cPickle
import time

import numpy
import scipy.sparse

import theano
import sys
from theano import tensor as T
from theano import sparse as S
from theano.tensor.shared_randomstreams import RandomStreams
from theano import ProfileMode
from theano.sparse.sandbox import sp as S1
from theano.sparse.sandbox import sp2 as S2


class DenoisingAE(object):
    """
    Denoising Auto-Encoder with sigmoid hiddens and sigmoid reconstruction.
    If the sampling parameters is smaller than 1 the input is assumed to be
    sparse and reconstruction sampling will be used.
    
    Parameters
    ----------
    
    n_inputs : int
        number of dimensions in the input
    n_hiddens : int
        number of dimensions in the hidden code
    W : array, shape = [n_inputs, n_hiddens], optional
        weight matrix for the model. If not provided a good default
        initialization will be used.
    c : array, shape = [n_hiddens], optional
        bias vector of the encoder. If not provided a good default
        initialization will be used.
    b : array, shape = [n_inputs], optional
        bias vector of the decoder. If not provided a good default
        initialization will be used.
    learning_rate : float, optional
        learning rate to use for gradient descent.
    noise : float, optional
        amount of binomial masking noise to use during learning. The noise
        is the probability that an individual unit will be masked.
    sampling : float optional
        If sampling = 1 then reconstruction sampling will not be used and
        the input is assumed to be a dense matrix. If sampling < 1 then
        reconstruction sampling will be used. The percentage of sampling
        should always be equal to the average number of non-zeros in the input.
    
    References
    ----------
    P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio and P.A. Manzagol,
    Stacked Denoising Autoencoders: Learning Useful Representations in a
    Deep Network with a Local Denoising Criterion, 
    Journal of Machine Learning Research, 11:3371--3408, 2010
    
    Y. Dauphin, X. Glorot, Y. Bengio. Large-Scale Learning of Embeddings with
    Reconstruction Sampling. In Proceedings of the 28th International Conference
    on Machine Learning (ICML 2011).
    """
    def __init__(self, input, n_inputs, n_hiddens, W=None, c=None, b=None, noise=0.1, sampling=0.1, learning_rate=0.1):
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.noise = noise
        self.sampling = sampling
        self.input = input
        self.learning_rate = learning_rate
        
        if W == None:
            nnz = n_inputs if sampling == 1 else int(sampling * n_inputs)
            W_values = numpy.asarray( numpy.random.uniform(
                low  = - numpy.sqrt(6./(nnz+n_hiddens)),
                high =   numpy.sqrt(6./(nnz+n_hiddens)),
                size = (n_inputs, n_hiddens)), dtype = theano.config.floatX)  # @UndefinedVariable
            W = theano.shared(W_values, borrow=True)

        if c is None:
            c = theano.shared(value=numpy.zeros(n_hiddens,
                                dtype=theano.config.floatX), name='c')  # @UndefinedVariable

        if b is None:
            b = theano.shared(value=numpy.zeros(n_inputs,
                                dtype=theano.config.floatX),name='b')  # @UndefinedVariable
        
        self.W = W
        self.c = c
        self.b = b
        self.params = [self.W, self.c, self.b]

        self.rng = RandomStreams(numpy.random.randint(2**30))
        
    
    def _encode(self, x, general_preprocessing_done=False):
        """
        Returns the hidden code for input x.
        
        Parameters
        ----------
        x : array, [n_samples, n_inputs]
        
        Returns
        -------
        h : array, [n_samples, n_hiddens]
        """
        if isinstance(x.type, S.SparseType):
            return T.nnet.sigmoid(S.dot(x, self.W) + self.c)
        else:
            return T.nnet.sigmoid(T.dot(x, self.W) + self.c)
    
    def _decode(self, h, p=None):
        """
        Returns the reconstruction based on hidden code h. If the sampling
        pattern p is provided reconstruction sampling will be used.
        
        Parameters
        ----------
        h : array, [n_samples, n_hiddens]
        p : array, [n_samples, n_inputs], optional
            binary sparse matrix indicating which inputs to reconstruct.
        
        Returns
        -------
        x : array, [n_samples, n_inputs]
            this matrix is dense if a sampling pattern p is not provided, and
            sparse otherwise.
        """
        if p == None:
            return T.nnet.sigmoid(T.dot(h, self.W.T) + self.b)
        else:
            a = S2.sampling_dot(h, self.W, p)
            return S2.structured_sigmoid(S2.structured_add_s_v(a, self.b))
    
    def _loss(self, x):
        """
        Returns the loss function of the auto-encoder for input x. If the
        input is sparse recontrusction sampling will automatically be used,
        otherwise, if dense, the normal cost will be used.
        
        Parameters
        ----------
        x : array, [n_samples, n_inputs]
        
        Returns
        -------
        loss : float,
        """
        if isinstance(x.type, S.SparseType):
            x_ = x * S2.csr_fbinomial(1, self.sampling, x.shape)
            h = self._encode(x_, general_preprocessing_done=True)
        
            p = S.sp_ones_like(x + S2.csr_fbinomial(1, self.sampling, x.shape))
            z = self._decode(h, p)
        
            r = -(x * S2.structured_log(z) + (p - x) * S2.structured_log(p - z))
            
            return (T.mean(S1.sp_sum(r, axis=1, sparse_grad=True)), z)
        else:
            x_ = x * self.rng.binomial(size=x.shape,
                n=1, p=(1 - self.noise), dtype=theano.config.floatX)  # @UndefinedVariable
            h = self._encode(x_, general_preprocessing_done=True)
            z = self._decode(h)
        
            return (T.sum(-(x*T.log(z) + (1.-x)*T.log(1.-z)), axis=1), z)
    
    def update(self):
        
        L, z = self._loss(self.input)
        cost = T.mean(L)
        gparams = T.grad(cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - self.learning_rate * gparam))
        print "\n"
        print self.W.eval()
        return (self.input, z, L, cost, updates)
    
    def save(self, tag=None):
        """
        Save the model to disk.
        
        Parameters
        ----------
        tag : str, optional
            optional tag to add to the filename to distinguish the files.
        """
        if tag == None:
            tag = ""
        else:
            tag = "_%s" % tag
        
        numpy.save("W%s.npy" % tag, self.W.get_value(borrow=True))
        numpy.save("c%s.npy" % tag, self.c.get_value(borrow=True))
        numpy.save("b%s.npy" % tag, self.b.get_value(borrow=True))
    
    
if __name__ == '__main__':
    training_epochs=10
    batch_size=2
#     train_set_x = numpy.array(numpy.random.randint(2, size = 500), dtype = theano.config.floatX).reshape((1, 500))  # @UndefinedVariable
    train_set_x = numpy.array(numpy.zeros(20), dtype = theano.config.floatX).reshape((2, 10))  # @UndefinedVariable
    train_set_x[0,0] = 1
    train_set_x[0,1] = 1
    train_set_x[1,2] = 1
    train_set_x[1,3] = 1
    train_set_x = theano.shared(train_set_x)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    print n_train_batches
    
    index = T.lscalar()  # index to a [mini]batch
    index = 0
    sampling = 0.01
    
    x = (T.matrix('x') if sampling in [1, None]
            else S.csr_matrix('x'))
    
    dae = DenoisingAE(input=train_set_x, sampling=sampling, n_inputs=10, n_hiddens=3, learning_rate=0.1)
    inputLayer, outputLayer, L, cost, updates = dae.update()
    train = theano.function(inputs = [], outputs = [inputLayer, outputLayer, L, cost], updates=updates, 
                            givens = {x: train_set_x})
    for i in range(1000):
#         inputLayer, outputLayer, L, cost, updates = dae.update()
        t = train()
        print "cost: " + str(t[3])
#     fit = theano.function([self.input], loss, updates=updates)
#     encode = theano.function([self.input], self._encode(self.input))
    print L.eval()
    print cost.eval()
    print inputLayer.eval()
    print outputLayer.eval()
    sys.exit(0)
    
    train = theano.function(inputs = [index], outputs = [inputLayer, outputLayer, L, cost], updates=updates, 
                            givens = {x: train_set_x[index * batch_size: (index + 1) * batch_size, :]})
    
    start_time = time.clock()
    
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


