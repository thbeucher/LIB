#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Thomas Beucher
Module: RBFN
Description: We find here functions which allow to compute a RBFN regression
'''

import numpy as np

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

class rbfn():
    
    def __init__(self, nbFeatures, inputDim, outputDim):
        '''
	Initializes class parameters
	
	Input:     -nbFeature: int, number of feature in order to perform the regression
        '''
        self.nbFeat = nbFeatures
        self.title = "rbfn"
        self.inputDimension = inputDim
        self.outputDimension = outputDim
        print("dimensions : " + str(self.inputDimension) + "x" +  str(self.outputDimension))
        self.theta = np.zeros((self.nbFeat, self.outputDimension))

    def setTheta(self, theta):
        self.theta = theta

    def loadTheta(self,thetaFile):
        self.theta = np.loadtxt(thetaFile)
        #print ("theta LOAD : ", self.theta)
        return self.theta

    def saveTheta(self,fileName):
        '''
        Records theta under numpy format
        
        Input:    -fileName: name of the file where theta will be recorded
              -theta: recorded theta
        '''
        #print ("theta SAVE:", self.theta)
        np.savetxt(fileName, self.theta)

    def setTrainingData(self, inputData, outputData):
        '''
        Verifies the validity of the given input and output data
        Data should be organized by columns
        
        Input:      -inputdata, numpy N-D array
                    -outputData, numpy N-D array
        '''
        self.inputData = inputData
        self.outputData = outputData

        #Getting input and output dimensions and number of samples
        numberOfInputSamples = len(inputData)
        numberOfOutputSamples = len(outputData)
        #check if there are the same number of samples for input and output data
        assert(numberOfInputSamples == numberOfOutputSamples), "Number of samples not equal for output and input"
        #check dimensions
        assert(len(inputData[0]) == self.inputDimension), "Mismatch in input dimension"
        assert(len(outputData[0]) == self.outputDimension), "Mismatch in output dimension"

        self.numberOfSamples = numberOfInputSamples
        self.setCentersAndWidths()
           
    def setCentersAndWidths(self):
        '''
        Sets the centers and widths of Gaussian features.
        Uses linspace to evenly distribute the features.
        '''
        #get max and min of the input data
        minInputData = np.min(self.inputData, axis = 0)
        maxInputData = np.max(self.inputData, axis = 0)
        rangeForEachDim = maxInputData - minInputData
        #set the sigmas
        widthConstant = 2.2*rangeForEachDim / self.nbFeat
        #create the diagonal matrix of sigmas to compute the gaussian
        self.widths = np.diag(widthConstant)
         #coef for Gaussian features
        self.norma = 1/np.sqrt(((2*np.pi)**self.inputDimension)*np.linalg.det(self.widths))
        #print ("RBFN : constante de normalisation : ", self.norma)
        self.invcovar = np.linalg.pinv(self.widths)
        linspaceForEachDim = []
        #set the number of gaussian used and allocate them in each dimensions
        for i in range(self.inputDimension):
            linspaceForEachDim.append(np.linspace(minInputData[i], maxInputData[i], self.nbFeat))
            #get matrix with all the possible combinations to find each centers
        self.centersInEachDimensions = cartesian(linspaceForEachDim)
        self.nbFeatures = len(self.centersInEachDimensions)
        print("nbfeatures:", self.nbFeatures)

    def train_rbfn(self):
        '''
        Training function to learn the approximation (i.e. regression)
        
        '''
        self.theta = []
        for i in range(self.outputDimension):
            K = []
            for val in self.inputData:
                 K.append(self.computeAllWeights(val))
            Kmat = np.matrix(K).T
            A = np.dot(Kmat, Kmat.T)
            inv = np.linalg.pinv(A)
            vec = self.outputData.T
            y = np.array(vec[i].T)
            b = np.dot(Kmat, y).T
            self.theta.append(np.dot(inv,b))
    
    def computeFeatureWeight(self, inputVal, gauss):
        '''
        Computes the value of an input with respect to one Gaussian feature
        
        Input:     -inputVal: one point in the input space (an input vector)
        
        Output:    -phi: a number: the value of phi(inputVal)
        '''
        xu = inputVal - gauss
        xus = np.dot(xu, self.invcovar)
        xg = np.dot(xus,xu.T)
        phi = self.norma*np.exp(-0.5*xg)
        return phi

    def computeAllWeights(self, inputVal):
        '''
        Computes the value of an input with respect to all Gaussian features
        
        Input:     -inputVal: one point in the input space (an input vector)
        
        Output:    -phi: a vector of values of all phi(x) for the input x and all phi
        '''
        retour = []
        for x in self.centersInEachDimensions:
            phi = self.computeFeatureWeight(inputVal, x)
            retour.append(phi)
        return retour

    def computeOutput(self, inputVal):
        '''
        Returns the output depending on the given input and theta
        
        Input:      -inputVal: numpy N-D array
                    -theta: numpy N-D array
        
        Output:     -fa_out: numpy N-D array, output approximated
        '''
        assert(inputVal.shape[0]==self.inputDimension), "RBFN: Bad input format"
        output = []
        for i in range(self.outputDimension):
            tmp = self.computeAllWeights(inputVal)
            output.append(np.dot(tmp, self.theta[i]))
        return output
    
        '''
        #if more than one sample
        else:
            for i in range(inputData.shape[0]):
                x_u = np.array([inputData[i]]).T - self.centersInEachDimensions.T
                x_u_s = np.dot(x_u.T, np.linalg.pinv(self.widths))
                x = x_u_s * (x_u.T)
                xf = np.sum(x, axis = 1)
                xfe = self.norma*np.exp(-0.5*xf)
                if i == 0:
                    phi = np.array([xfe]).T
                else:
                    phi = np.hstack((phi, np.array([xfe]).T))
        '''
