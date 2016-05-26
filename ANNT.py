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
import theano
import theano.tensor as T


#activation functions
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
