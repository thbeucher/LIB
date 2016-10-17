#-------------------------------------------------------------------------------
# Name:        theanoL
# Purpose:
#
# Author:      tbeucher
#
# Created:     14/10/2016
# Copyright:   (c) tbeucher 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import theano.tensor as T
import theano as th


#adding two scalars
def addTwoScalars(a, b):
    #define symbols ie variables
    #dscalar = floating-point scalar
    x = T.dscalar('x')
    y = T.dscalar('y')
    #operation we want to perform
    z = x + y
    #create function taking x and y as inputs and giving z as output
    #first arg is list of variables that will be provided as inputs
    #second arg is a single or a list of variables, it is what we want to see as output
    f = th.function([x, y], z)
    print(f(a, b))

#addTwoScalars(2, 3)

#adding two matrices
def addTwoMatrices(a, b):
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    z = x + y
    f = th.function([x, y], z)
    print(f(a, b))

#addTwoMatrices([[1,2], [3,4]], [[10,20], [30,40]])

#exemple
def ex1(a, b):
    x = T.vector('x')
    y = T.vector('y')
    z = x**2 + y**2 + 2*x*y
    f = th.function([x, y], z)
    print(f(a, b))

#ex1([1,2,3], [10,20,30])


#apply function to a matrix element-wise
#here the logistic function
def applyLogistic(a):
    x = T.dmatrix('x')
    out = 1/(1+T.exp(-x))
    logi = th.function([x], out)
    print(logi(a))

#applyLogistic([[0,1], [-1,-2]])

#compute more than one thing at the same time
def multipleThingAtTheSameTime(a, b):
    x, y = T.dmatrices('x', 'y')
    diff = x - y
    abs_diff = abs(diff)
    diff_squared = diff**2
    summ = x + y
    f = th.function([x,y], [diff, abs_diff, diff_squared, summ])
    print(f(a, b))

#multipleThingAtTheSameTime([[1,2], [3,4]], [[2,3], [4,5]])

#default value for an argument
def defaultValue(*arg):
    x, y, w = T.dscalars('x', 'y', 'w')
    z = x + y + w
    f = th.function([x, th.In(y, value=1), th.In(w, value=2, name='wName')], z)
    if len(arg) == 3:
        print(f(arg[0], wName = arg[1], y = arg[2]))
    elif len(arg) == 2:
        print(f(arg[0], arg[1]))
    else:
        print(f(arg[0]))

#defaultValue(10)
#defaultValue(10, 20)
#defaultValue(10, 20, 30)


#shared variables
def accumulator(a):
    #shared variable can be use by more than one function
    state = th.shared(0)
    inc = T.iscalar('inc')
    #update must be a list of pairs of the form (shared-variable, new expression)
    #or a dictionary whose keys are shared-variables and values are the new expressions
    acc = th.function([inc], state, updates=[(state, state+inc)])
    for el in a:
        print(str(el) + " --> " + str(acc(el)))
    #we can acces to the state value with state.get_value() and state.set_value()

#accumulator([1, 10, 200, 10])

#if we have defined a formula using a specific shared variable but we don't
#want to use its value --> we can use givens parameter
def replaceSharedVariable(a, b):
    state = th.shared(0)
    inc = T.iscalar('inc')
    #formula defined using state shared-variable
    fn = state * 2 + inc
    #the type of the argument used to replace the previous shared variable
    #, here state, must match
    foo = T.scalar(dtype=state.dtype)
    dec = th.function([inc, foo], fn, givens=[(state, foo)])
    print(dec(a, b))

#replaceSharedVariable(1,3)

#derivatives
#compute gradient
def gradient(a):
    x = T.dscalar('x')
    y = x**2
    z = 1/x
    gy = T.grad(y, x)
    gz = T.grad(z, x)
    print(th.pp(gy))
    print(th.pp(gz))
    f = th.function([x], gy)
    g = th.function([x], gz)
    print(f(a))
    print(g(a))

#gradient(2)


#conditions
#ifelse vs switch


#loop
#scan function
#unchanging variables are passed to scan as non_sequences
#initialization occurs in outputs_info

#compute A**k elementwise
def aPowerK(a, b):
    k = T.iscalar('k')
    A = T.vector('A')
    #order of the parameters
    #the output of the prior call to fn (or the initial value, initially) is
    #the first parameter, followed by all non-sequences
    result, updates = th.scan(fn=lambda prior_result, A: prior_result * A,\
                                outputs_info=T.ones_like(A), non_sequences=A, n_steps=k)
    #if we juste want the final result and let the intermediate values be discarded
    final_result = result[-1]

    power = th.function(inputs=[A,k], outputs=final_result, updates=updates)
    print(power(a,b))

#aPowerK(range(10), 2)

#calculating a polynomial
def calculPoly(a, b):
    x = T.scalar('x')
    coefs = T.vector('coefs')
    max_coef_supported = 10000
    #generate the components of the polynomial
    #outputs_info to None indicates to scan that it doesn't need to pass the prior result to fn
    #there is no accumulation of results
    #use of theano.tensor.arange to simulate python's enumerate
    components, updates = th.scan(fn=lambda coef, power, free_var: coef*(free_var**power),\
                                    outputs_info=None, sequences=[coefs, T.arange(max_coef_supported)],\
                                    non_sequences=x)
    #sum them up
    poly = components.sum()

    calculate_poly = th.function(inputs=[coefs, x], outputs=poly)
    print(calculate_poly(a, b))

    #to improve memory usage we can accumulate coefficients along the way and then take the last one
    def accu(coef, power, prior_val, free_var):
        return prior_val + (coef*(free_var**power))
    out = T.as_tensor_variable(np.asarray(0, coefs.dtype))
    result, updates2 = th.scan(fn=accu, outputs_info=out,\
                                 sequences=[coefs, T.arange(max_coef_supported)],\
                                 non_sequences=x)
    final_result = result[-1]
    calculate_poly2 = th.function(inputs=[coefs, x], outputs=final_result, updates=updates2)
    print(calculate_poly2(a,b))
    #not sure but the order for variables are
    #sequences, output, non_sequences

#calculPoly([1,0,2], 3)

#simple accumulation into a scalar, ditching lambda
def ditchingLambda(a):
    up_to = T.iscalar('up_to')
    seq = T.arange(up_to)
    def acc(val_arange, sum_actual_value):
        return sum_actual_value + val_arange
    out = T.as_tensor_variable(np.asarray(0, seq.dtype))
    result, updates = th.scan(fn=acc, outputs_info=out, sequences=seq)
    triangular_seq = th.function(inputs=[up_to], outputs=result)
    print(triangular_seq(a))

#ditchingLambda(15)


