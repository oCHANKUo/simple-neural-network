# this is a 3 layer neural network
# why 3 layers? 1st layer will combine the inputs, the 2nd will map them to the output and uses the 1st layer output as the inputs

import numpy as np

def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    
    return 1/(1+np.exp(-x))  #the same Sigmoid function as before

X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

# randomly intialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in range(60000):

    # Feed forward through layers 0, 1 and 2
    l0 = X
    l1 = nonlin(np.dot(l0, syn0)) #hidden layer
    l2 = nonlin(np.dot(l1, syn1))

    # how much did we miss the target value?
    l2_error = y - l2

    if (j% 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(l2_error))))

    # in what direction is the target value?
    # were we really sure? if so, dont change too much
    l2_delta = l2_error*nonlin(l2, deriv=True)

    #how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    # were we really sure, if so dont change too much
    l1_delta = l1_error * nonlin(l1, deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)