#-------------------------------------------------------------------------------
# Name:        Artificial neural network Theano
# Purpose:     Artificiam neural network structure using theano
#
# Author:      tbeucher
#
# Created:     23/03/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import lasagne
import theano
import theano.tensor as T
from theano.tensor import shared_randomstreams
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax

#activation functions
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


def dropout_layer(layer, p_dropout):
    '''
    returns a dropout layer

    input: -layer - tensor
           -p_dropout - % of dropout neuron

    output: -the dropout layer

    '''
    #equivalent of randomState for theano
    srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
    #create a binary mask
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    #we work with float32 but int*float32 = float64 so we have to cast the result
    return layer*T.cast(mask, theano.config.floatX)

def input_layer(shape):
    '''
    shape: (None, num_frames, input_width, input_height)

    '''
    return lasagne.layers.InputLayer(shape=shape)


class FullyConnectedLayer(object):

    def __init__(self, obj, choice="theano"):
        '''
        obj - python list:
                if theano:
                    -n_in:
                    -n_out:
                    -activation_fn: activation function
                    -p_dropout:
                if denselayer:
                    -layer_in: previous layer
                    -nb_units: x - number of neuron
                    -nonlinearity: see lasagne.nonlinearities, default to rectify
                    -init_w: see lasagne.init, nature=HeUniform()
                    -init_b: see lasagne.init, nature=Constant(.1)
        choice - type of initialization:
                -theano: classic implementation with theano
                -denselayer: use of lasagne.layers.DenseLayer

        '''
        if choice == "theano":
            self.n_in, self.n_out, self.activation_fn, self.p_dropout = obj
            # Initialize weights and biases
            self.w = theano.shared(
                np.asarray(
                    np.random.normal(
                        loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                    dtype=theano.config.floatX),
                name='w', borrow=True)
            self.b = theano.shared(
                np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                           dtype=theano.config.floatX),
                name='b', borrow=True)
            self.params = [self.w, self.b]
        elif choice == "denselayer":
            l_in, nb_units, nonlinearity, w, b = obj
            self.layer = lasagne.layers.DenseLayer(l_in, num_units=nb_units,
                                              nonlinearity=nonlinearity, w=w, b=b)

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

class ConvPoolLayer(object):
    '''
    Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    '''
    def __init__(self, filter_shape, image_shape, poolsize=(2, 2), activation_fn=sigmoid):
        '''
        `filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.
        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.
        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        '''
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers

class ConvolutionalLayer():

    def __init__(self, obj, choice="theano"):
        '''
        obj - python list:
                if theano:
                    -filter_shape: (nb of filters, nb of input feature maps, filter height, filter width)
                    -image_shape: (mini-batch size, nb of input feature maps, image heigth, image width)
                    -activation_fn: activation function, default to sigmoid
                if cuda or dnn:
                    -layer_in: previous layer
                    -nb_filters: x - number of filters
                    -filter_size: (y, x) - size of the filter
                    -stride: (y, x)
                    -nonlinearity: see lasagne.nonlinearities, default to rectify
                    -init_w: see lasagne.init, nature=HeUniform()
                    -init_b: see lasagne.init, nature=Constant(.1)
                    -dimshuffle: True or False
                if lasagne_simple: same as cuda or dnn but without dimshuffle
        choice - type of initialization:
                -theano: classic implementation with theano
                -cuda: use of lasagne.layers.cuda_convnet
                -pylearn: use of lasagne.layers.dnn

        '''
        if choice == "theano":
            self.filter_shape, self.image_shape, self.activation_fn = obj
            n_out = filter_shape[0]*np.prod(filter_shape[2:])
            self.w = theano.shared(
                np.asarray(
                    np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                    dtype=theano.config.floatX),
                borrow=True)
            self.b = theano.shared(
                np.asarray(
                    np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                    dtype=theano.config.floatX),
                borrow=True)
            self.params = [self.w, self.b]
        elif choice == "lasagne_simple":
            l_in, num_filters, filter_size, stride, nonlinearity, w, b = obj
            self.layer = lasagne.layers.Conv2DLayer(l_in, num_filters=num_filters,
                                                    filter_size=filter_size,
                                                    stride=stride,
                                                    w=w, b=b,
                                                    nonlinearity=nonlinearity)
        #requires GPU
        elif choice == "cuda":
            l_in, num_filters, filter_size, stride, nonlinearity, w, b, dimshuffle = obj
            from lasagne.layers import cuda_convnet
            self.layer = cuda_convnet.Conv2DCCLayer(l_in,
                                              num_filters=num_filters,
                                              filter_size=filter_size,
                                              stride=stride,
                                              nonlinearity=nonlinearity,
                                              w=w, b=b, dimshuffle=dimshuffle)
        #requires GPU
        elif choice == "dnn":
            l_in, num_filters, filter_size, stride, nonlinearity, w, b = obj
            from lasagne.layers import dnn
            self.layer = dnn.Conv2DDNNLayer(l_in, num_filters=num_filters,
                                      filter_size=filter_size, stride=stride,
                                      nonlinearity=nonlinearity,
                                      w=w, b=b)

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(input=self.inpt, filters=self.w,
                               filter_shape=self.filter_shape,
                               image_shape=self.image_shape)
        self.output = self.activation_fn(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        #no dropout in the convolutional layers
        self.output_dropout = self.output



class Network():

    def __init__(self, obj, choice="theano"):
        '''
        obj - python list:
            if choice = theano:
                -layers: python list containing the network architecture
                -mini_batch_size
            if choice == lasagne:

        choice - type of initialization:
            theano or lasagne

        '''
        if choice == "theano":
            self.layers, self.mini_batch_size = obj
            self.params = [param for layer in self.layers for param in layer.params]
            self.x = T.matrix("x")
            self.y = T.ivector("y")
            init_layer = self.layers[0]
            init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
            for j in xrange(1, len(self.layers)):
                prev_layer, layer  = self.layers[j-1], self.layers[j]
                layer.set_inpt(prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
            self.output = self.layers[-1].output
            self.output_dropout = self.layers[-1].output_dropout
        elif choice == "lasagne":
            a=1
        #for DQN
        #initialize replay memory D
        #initialize action-value function Q with random weights
        #observe initial state s

    def DQN(self):
        '''

        '''
        #repeat
            #select an action a
                #with probability eps select a random action
                #otherwise select a = argmax-a'-Q(s,a')
            #carry out action a
            #observe reward r and new state s'
            #store experience <s,a,r,s'> in replay memory D

            #sample random transitions <ss,aa,rr,ss'> from replay memory D
            #calculate target for each minibatch transition
                #if ss' is terminal state then tt = rr
                #otherwise tt == rr + gamma*max-a'-Q(ss', aa')
            #train the Q network using (tt - Q(ss,aa))^2 as loss

            #s = s'
        #until terminated
