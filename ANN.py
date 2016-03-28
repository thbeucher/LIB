#-------------------------------------------------------------------------------
# Name:        Artificial neural network
# Purpose:
#
# Author:      tbeucher
#
# Created:     23/03/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np



class MLP:
    '''
    Implements a multilayer perceptron

    '''

    def __init__(self, networkStruct):
        '''
        Inits the neural network

        Input: networkStructure - python list - give the number of neuron for each layers
        
        '''
        self.nbLayer = len(networkStruct)
        self.networkStruct = networkStruct
        self.basicInit()

    def basicInit(self):
        '''
        Random initializes of biaises and weights using gaussian distribution with mean 0 and standard deviation 1

        '''    
        self.weights = [np.random.randn(nbNeu, nbIn) for nbNeu, nbIn in zip(self.networkStruct[1:], self.networkStruct[:-1])]
        self.biases = [np.random.randn(nbNeu,1) for nbNeu in self.networkStruct[1:]]

    def feedForward(self, a):
        '''
        Computes the output of the network for an input a

        Input: a - numpy array or numpy list - input for the network

        Output: a - numpy array - output of the network

        '''
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def update(self, batch, eta):
        '''
        Update the network's weights and biases by applying gradient descend using backpropagation

        Input: batch - python list - list of tuples (x,y) where x in the input and y the desired output
               eta - float - learning rate
  
        '''
        nablaW = [np.zeros(w.shape) for w in self.weights]
        nablaB = [np.zeros(b.shape) for b in self.biases]
        for x, y in batch:
            deltaW, deltaB = self.backProp(x, y)
            nablaW = [nw+dnw for nw, dnw in zip(nablaW, deltaW)]
            nablaB = [nb+dnb for nb, dnb in zip(nablaB, deltaB)]
        self.weights = [w-(eta/len(batch))*nw for w, nw in zip(self.weights, nablaW)]
        self.biases = [b-(eta/len(batch))*nb for b, nb in zip(self.biases, nablaB)]

    def backProp(self, x, y):
        '''
        Computes the gradient descent

        Input: x - input
               y - desired output

        Output: 

        '''
        deltaW = [np.zeros(w.shape) for w in self.weigths]
        deltaB = [np.zeros(b.shape) for b in self.biases]
        #feedForward
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        #output error
        delta = (activations[-1] - y)*self.sigmoidPrime(zs[-1])
        deltaW[-1] = np.dot(delta, activation[-2].T)
        deltaB[-1] = delta
        #backpropagate the error
        for l in xrange(2, self.nbLayer):
            z = zs[-1]
            sp = self.sigmoidPrime(z)
            delta = np.dot(self.weights[-l+1].T, delta)*sp
            deltaW[-l] = np.dot(delta, activations[-l-1].T)
            deltaB[-l] = delta
        return deltaW, deltaB

    def sigmoid(self, a):
        '''
        Input:
            a - numpy array

        Output:
            return the value of a throught the sigmoid function

        '''
        return 1/(1 + np.exp(-a))

    def sigmoidPrime(self, a):
        '''
        Input:
            a - numpy array

        Output:
            return the value of a throught the derivative of the sigmoid function

        '''
        return self.sigmoid(a)*(1-self.sigmoid(a))

    def hyperbolicTan(self, a):
        '''
        Input:
            a - numpy array

        Output:
            return the value of a throught the tanh function

        '''
        return np.tanh(a)


