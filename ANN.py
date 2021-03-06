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
import json


def sigmoid(a):
    '''
    Input:
        a - numpy array

    Output:
        return the value of a throught the sigmoid function

    '''
    return 1/(1 + np.exp(-a))

def sigmoidPrime(a):
    '''
    Input:
        a - numpy array

    Output:
        return the value of a throught the derivative of the sigmoid function

    '''
    return self.sigmoid(a)*(1-self.sigmoid(a))

def hyperbolicTan(a):
    '''
    Input:
        a - numpy array

    Output:
        return the value of a throught the tanh function

    '''
    return np.tanh(a)

class QuadraticCost():

    @staticmethod
    def fn(a, y):
        '''
        Return the cost

        Input:
            a - output
            y -  desired output

        Output:

        '''
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        '''
        Return the error delta from the output layer

        Input:
            a - output
            y -  desired output
            z - wa + b

        Output:

        '''
        return (a-y) * sigmoid_prime(z)

class CrossEntropyCost():

    @staticmethod
    def fn(a, y):
        '''
        Return the cost - np.nan_to_num ensure numerical stability

        Input:
            a - output
            y -  desired output

        Output:

        '''
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        '''
        Return the error delta from the output layer

        Input:
            a - output
            y -  desired output

        Output:

        '''
        return (a-y)

class MLP:
    '''
    Implements a multilayer perceptron

    '''

    def __init__(self, networkStruct, cost = CrossEntropyCost):
        '''
        Inits the neural network

        Input: networkStructure - python list - give the number of neuron for each layers

        '''
        self.nbLayer = len(networkStruct)
        self.networkStruct = networkStruct
        self.cost = cost
        self.specialInit()

    def basicInit(self):
        '''
        Random initializes of biaises and weights using gaussian distribution with mean 0
        and standard deviation 1

        '''
        self.weights = [np.random.randn(nbNeu, nbIn) for nbNeu, nbIn in zip(self.networkStruct[1:], self.networkStruct[:-1])]
        self.biases = [np.random.randn(nbNeu,1) for nbNeu in self.networkStruct[1:]]

    def specialInit(self):
        '''
        Random initializes of biaises and weights using gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of weights connecting
        to the same neuron

        '''
        self.weights = [np.random.randn(nbNeu, nbIn)/np.sqrt(nbIn) for nbNeu, nbIn in zip(self.networkStruct[1:], self.networkStruct[:-1])]
        self.biases = [np.random.randn(nbNeu,1) for nbNeu in self.networkStruct[1:]]

    def feedForward(self, a):
        '''
        Computes the output of the network for an input a

        Input: a - numpy array or numpy list - input for the network

        Output: a - numpy array - output of the network

        '''
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
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
            activation = sigmoid(z)
            activations.append(activation)
        #output error
        delta = self.cost.delta(zs[-1], activations[-1], y)
        deltaW[-1] = np.dot(delta, activation[-2].T)
        deltaB[-1] = delta
        #backpropagate the error
        for l in xrange(2, self.nbLayer):
            z = zs[-1]
            sp = sigmoidPrime(z)
            delta = np.dot(self.weights[-l+1].T, delta)*sp
            deltaW[-l] = np.dot(delta, activations[-l-1].T)
            deltaB[-l] = delta
        return deltaW, deltaB

    def save(self, fileName):
        '''
        Save the neural network to the file fileName

        '''
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


