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
