"""
"""
import cPickle
import gzip
import os
import sys

import numpy

import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.sandbox as sand
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from grbm import GRBM
from rbm import RBM


class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """
        
        #self.forward_result = None
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.best_params = None
        self.last_params = None
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  
        self.y = T.ivector('y') 


        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params.extend(sigmoid_layer.params)
            #self.last_params.extend(sigmoid_layer.last_params)
            # Construct an RBM that shared weights with this layer
            if (i == 0):
                rbm_layer = GRBM(numpy_rng=numpy_rng,
                                 theano_rng=theano_rng,
                                 input=layer_input,
                                 n_visible=input_size,
                                 n_hidden=hidden_layers_sizes[i],
                                 W=sigmoid_layer.W,
                                 hbias=sigmoid_layer.b)
            else:
                rbm_layer = RBM(numpy_rng=numpy_rng,
                                theano_rng=theano_rng,
                                input=layer_input,
                                n_visible=input_size,
                                n_hidden=hidden_layers_sizes[i],
                                W=sigmoid_layer.W,
                                hbias=sigmoid_layer.b)
    
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)
        #self.last_params.extend(self.logLayer.last_params)
        
        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)
    
    def save_best_params(self):
        self.best_params = []
        for param in self.params:
            self.best_params.append(theano.shared(numpy.array(param.get_value(borrow=False),dtype=theano.config.floatX)))
    
    def load_best_params(self):
        if self.params is not None:
            for i in xrange(len(self.best_params)):
                self.params[i].set_value(self.best_params[i].get_value())

    def save_last_params(self):
        self.last_params = []
        for param in self.params:
            self.last_params.append(theano.shared(numpy.array(param.get_value(borrow=False),dtype=theano.config.floatX)))
    
    def load_last_params(self):
        if self.params is not None:
            for i in xrange(len(self.last_params)):
                self.params[i].set_value(self.last_params[i].get_value())


    
 #   def rewind_params(self):
 #       tmp = self.params
 #       self.params = self.last_params
 #       self.last_params = tmp
 #
  
    def pretraining_functions(self, train_set_x, batch_size, k, weight_p=0.0002):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use
        momentum      = T.scalar('momentum')
        
        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        n = 0
        
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            
            cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                                 persistent=None, 
                                                 k=k,
                                                 moment=momentum,
                                                 batch_sz=batch_size,
                                                 weight_penalty=weight_p)
            # compile the theano function
            fn = theano.function(inputs=[index,learning_rate,theano.Param(momentum, default=0.9)],
                                 outputs=T.cast(cost, theano.config.floatX),#theano.Out(sand.cuda.basic_ops.gpu_from_host(T.cast(cost, theano.config.floatX)), borrow=True),
                                 updates=updates,
                                 givens={self.x:
                                             train_set_x[batch_begin:batch_end]},
                                 name='pretrain '+str(n))
            # append `fn` to the list of functions
            pretrain_fns.append(fn)
            n += 1
            
        return pretrain_fns

#    def get_forward_result(self,idx):
#        if (self.forward_result is not None):
#            return self.forward_result[idx]
#        else: 
#            return None
        
    def compute_forward(self, features, labels, batch_sz, binary=True):
       
        n_batches = features.get_value(borrow=True).shape[0]/batch_sz
        #n_batches /= batch_size
        index = T.lscalar('index')  # index to a [mini]batch
        
        outputs = []
        
        if binary:
            for i in xrange(self.n_layers):
                outputs.append(self.rbm_layers[i].sample_h())
        else:
            for i in xrange(self.n_layers):
                outputs.append(self.sigmoid_layers[i].output)
        
        #for i in xrange(self.n_layers):
         #   outputs.append(self.sigmoid_layers[i].output)
        
        #outputs.append(T.cast(self.logLayer.p_y_given_x,theano.config.floatX))
        outputs.append(T.cast(self.errors, theano.config.floatX))

        forward_fn = theano.function(inputs=[index], outputs=outputs,
                                     givens={self.x: features[index*batch_sz:
                                                                  (index+1)*batch_sz],
                                             self.y: labels[index*batch_sz:
                                                                (index+1)*batch_sz]},
                                     name='forward_function')
        def compute_forward():
            return [forward_fn(i) for i in xrange(n_batches)]
            
        return compute_forward()
        
        
    def build_finetune_functions(self, datasets, batch_size_tr, batch_size_dev,batch_size_tst, weight_penalty=0.0002):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y)   = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size_dev
        
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size_tst

        

        index = T.lscalar('index')  # index to a [mini]batch
        lr_ft = T.scalar('lr')
        momentum = T.scalar('momentum')
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)
        
        # compute list of fine-tuning updates
        updates = []
        cnt = 0
        
        #for i in xrange(len(self.params)):
        #    updates.append((self.last_params[i],self.params[i]))
        
        for param, gparam in zip(self.params, gparams):
            if cnt%2 == 0: #it's  weights
                last_delta_W = theano.shared(numpy.array(param.get_value(), dtype=theano.config.floatX))
                delta = -(gparam + weight_penalty * param )* T.cast(lr_ft , theano.config.floatX)
                updates.append((param, param + (1-momentum)*delta + momentum*last_delta_W ))
                updates.append((last_delta_W , T.cast(delta,theano.config.floatX)))
                
            else: # it's bias 
                updates.append((param, param - gparam * T.cast(lr_ft , theano.config.floatX)))
            cnt+=1
        
        train_fn = theano.function(name='finetune_train',inputs=[index,lr_ft,theano.Param(momentum, default=0.9)],
                                   outputs=theano.Out(sand.cuda.basic_ops.gpu_from_host(T.cast(self.finetune_cost, theano.config.floatX))
                                                      ,borrow=True),
                                   updates=updates,
                                   givens={self.x: train_set_x[index * batch_size_tr:
                                                                   (index + 1) * batch_size_tr],
                                           self.y: train_set_y[index * batch_size_tr:
                                                                   (index + 1) * batch_size_tr]})
        
        test_score_i = theano.function([index], T.cast(self.errors, theano.config.floatX),
                 givens={self.x: test_set_x[index * batch_size_tst:
                                            (index + 1) * batch_size_tst],
                         self.y: test_set_y[index * batch_size_tst:
                                            (index + 1) * batch_size_tst]},
                                       name='finetune_test')
               
        
        valid_score_i = theano.function([index], T.cast(self.errors, theano.config.floatX),
                                        givens={self.x: valid_set_x[index * batch_size_dev:
                                                                        (index + 1) * batch_size_dev],
                                                self.y: valid_set_y[index * batch_size_dev:
                                                                        (index + 1) * batch_size_dev]},
                                        name='finetune_valid')

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score


    #def pretrain(self, train_set, pretraining_epochs=100, pretrain_lr=0.01, k=1, batch_size=10):
        
        

def test_DBN(finetune_lr=0.1, pretraining_epochs=100,
             pretrain_lr=0.01, k=1, training_epochs=1000,
             dataset='../data/mnist.pkl.gz', batch_size=10):
    """
    Demonstrates how to train and test a Deep Belief Network.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
    :type k: int
    :param k: number of Gibbs steps in CD/PCD
    :type training_epochs: int
    :param training_epochs: maximal number of iterations ot run the optimizer
    :type dataset: string
    :param dataset: path the the pickled dataset
    :type batch_size: int
    :param batch_size: the size of a minibatch
    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=28 * 28,
              hidden_layers_sizes=[1000, 1000, 1000],
              n_outs=10)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    for i in xrange(dbn.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)

    print '... finetunning the model'
    # early-stopping parameters
    patience = 4 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))


#if __name__ == '__main__':
    #test_DBN()
