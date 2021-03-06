#-------------------------------------------------------------------------------
# Name:        Utils
# Purpose:
#
# Author:      tbeucher
#
# Created:     21/03/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import shutil
import numpy as np
import json
from PIL import ImageGrab

def zipData(zipFileName, pathToZip):
    '''
    zip the given data

    Inputs:
        zipFileName - string - path name for the created zip file
        pathToZip - string - complete path for the folder to zip
    '''
    shutil.make_archive(zipFileName, 'zip', pathToZip)

def totuple(a):
    '''
    try to convert input into tuple

    Input:
        a - object to convert into tuple
    Ouput:
        tuple if success
    '''
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def clamp(var, mini, maxi):
    '''
    Ensures that var is between mini and maxi

    Inputs:
        var - number - variable to clamp between max and min given
        mini - number - the minimum value that var could take
        maxi - number - the maximum value that var could take

    Ouput:
        var - number - clamped variable

    '''
    if var < mini:
        var = mini
    if var > maxi:
        var = maxi
    return var

def readParams(pathToParam):
    '''
    Reads a parameters file and set data into a python dictionary

    Input:
        pathToParam - String - path to the parameters file

    Output:
        p - python dictionary - key = name of the parameter, value = value of the parameter
    '''
    with open(pathToParam, 'r') as file:
        alls = file.readlines()

    p = {}
    for line in alls:
        a = line.split(":")
        if a[0] == "nonbhl":
            t = a[1].split(",")
            t = [int(el) for el in t]
            p[a[0]] = t
        else:
            p[a[0]] = float(a[1])

    return p

def loadNetwork(fileName, networkObj):
    '''
    Load a neural network from the file filename - Returns an
    instance of Network

    '''
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = networkObj(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#look at getwindowdc getpixel to maybe speed up screenshot
def getImage(canv, ret = "np", conv = "grayscale"):
    '''
    Takes a screenshot of the canvas screen, convert to grayscale
    and convert in numpy array

    Import needed: -from PIL import ImageGrab
                   -import numpy as np

    Input: canv - tkinter canvas object
           ret - 'np' or 'all' to get in return:
               np: default value - only numpy array - pix
               all: numpy array, rgb and grayscale image - pix, snapshot and snapToGray
           conv - 'rgb' or 'grayscale' - choose image format
               grayscale: default value

    Ouput: pix - store image pixels in a numpy array
           snapshot - rgb pil image
           snapeToGray - grayscale image

    '''
    #get the coordinate of canvas in the windows screen
    x1, y1 = canv.winfo_rootx(), canv.winfo_rooty()
    #get height and width of the canvas
    h, w = canv.winfo_height(), canv.winfo_width()
    coordSnap = (x1, y1, x1+h, y1+w)
    #take a screenshot ie rgb image
    snapshot = ImageGrab.grab(coordSnap)
    if conv == "rgb":
        #convert image to array
        pix = np.array(snapshot)
    else:
        #convert rgb to grayscale
        snapToGray = snapshot.convert('L')
        #convert image to array
        pix = np.array(snapToGray)
        if ret == "all":
            return pix, snapshot, snapToGray
    if ret == "np":
        return pix


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

def list_to_dict(in_list):
    return dict((i, in_list[i]) for i in range(0, len(in_list)))


def exchange_key_value(in_dict):
    return dict((in_dict[i], i) for i in in_dict)
