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

    def __init__(self, NumberOfInputs, NumberOfOutputs, NumberOfHiddenLayers, NumberOfNeuronsByHiddenLayer, activationFunction):
        '''
        Creates a neural network and the weight matrix associated

        Inputs:
            NumberOfInputs - int
            NumberOfOutputs - int
            NumberOfHiddenLayers - int
            NumberOfNeuronsByHiddenLayer - python list - give the number of neurons for each hidden layer
            activationFunction - int - 0 = sigmoide / 1 = tanh

        '''
        assert (len(NumberOfNeuronsByHiddenLayer) == NumberOfHiddenLayers), "len of NumberOfNeuronsByHiddenLayer list must be equal to NumberOfHiddenLayers"
        self.noi = NumberOfInputs
        self.activationFunctionChoosed = activationFunction
        self.listOfActivationFunctions = [self.sigmoidActivation, self.hyperbolicTanActivation]
        self.neuralNet(NumberOfInputs, NumberOfOutputs, NumberOfHiddenLayers, NumberOfNeuronsByHiddenLayer)

    def neuralNet(self, noi, noo, nohl, nonbhl):
        '''
        Creates the artificial neural network

        for each layer, a matrix numberOfInput*numberOfNeuron is created and also a vector of bias for each neuron
        '''
        ANN = []
        #create first hidden layer
        ANN.append((np.random.uniform(-1, 1, (noi, nonbhl[0])), np.random.uniform(-1, 1, (1, nonbhl[0]))))
        #create hidden layer
        if nohl > 1:
            for i in range(nohl - 1):
                i += 1
                ANN.append((np.random.uniform(-1, 1, (nonbhl[i-1], nonbhl[i])), np.random.uniform(-1, 1, (1, nonbhl[i]))))
        #create output layer
        ANN.append((np.random.uniform(-1, 1, (nonbhl[nohl-1], noo)), np.random.uniform(-1, 1, (1, noo))))
        self.ANN = ANN

    def sigmoidActivation(self, a):
        '''
        Input:
            a - numpy array

        Output:
            return the value of a throught the sigmoid function

        '''
        return 1/(1 + np.exp(-a))

    def hyperbolicTanActivation(self, a):
        '''
        Input:
            a - numpy array

        Output:
            return the value of a throught the tanh function

        '''
        return np.tanh(a)

    def update(self, inputData):
        '''
        Input:
            inputData - numpy array - shape must be (1, numberOfInput) or (numberOfInput,)
        Output:
            outputData - numpy array, (1, numberOfOutputs)

        '''
        assert (np.asarray(inputData).size == self.noi), "The dimension of the inputData array is not correct!"
        for layer in self.ANN:
            #Compute input time weights
            res = np.dot(inputData, layer[0])
            #add in the bias
            resBias = res + layer[1]
            #Computes the output
            acti = self.listOfActivationFunctions[self.activationFunctionChoosed](resBias)
            inputData = acti
        return inputData