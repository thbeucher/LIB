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
