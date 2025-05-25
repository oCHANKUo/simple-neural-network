import numpy as np  #i can refer to numpy as np

# sigmoid function
# Sigmoid function takes any real number between positive & negative infinity,
# and sort of relates them to a value between 1 and 0. Or maps them to a value between 0 and 1
# We use it to convert numbers to probabilities
# sigmoid(x) = 1/(1+eulers number to the power -x)
# sigmoid(x) derivative = sigmoid(x)*(1-sigmoid(x))

# This function takes an x variable, and whether you want to calculate derivative
# or not, with the default being NO. 
def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x) # If the input is TRUE, it calculates the derivative. But we need to input the sigmoid value as the x input.
    return 1/(1+np.exp(-x))

# Input Dataset
x = np.array([ [0,0,1], 
               [0,1,1], 
               [1,0,1], 
               [1,1,1] ])

# Output Dataset
y = np.array([[0,1,1,0]]).T # .T means transpose, resulting in a column vector. So, 0,0,1 output is 0, and so on.

# Seed random numbers to make calculations. Seed is set to 1, to make the results reproducable
np.random.seed(1)

# Initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1 # Creates a 3*1 matrix with random values between 1 and 0. Then multiplies by 2 and substracts 1

for iter in range(1000): # Loop iterates 10000 times to train the nerwork

    # forward propagation
    l0 = x  # Input Layer 0 specified by the input data x
    l1 = nonlin(np.dot(l0, syn0))  # Output layer, np.dot() multiplies inputs by weights
    # Synapse0, first layer of weights, Connecting l0 and l1.

    # how much did we miss?
    l1_error = y - l1  # error = actual_output - predicted_output

    # Checking l1 ater the first iteration for comparison
    #if iter == 100000:
    #    print("After specific iterations:")
    #    print(l1)
    #    print("Error:")
    #    print(l1_error)
    #    print("Error Delta:")
    #    print(l1_delta)

    # multiply how much we missed by the slope of the sigmoid at values in l1
    l1_delta = l1_error * nonlin(l1, True)  # Multiplies the error by sigmoid derivative

    # update weights
    syn0 += np.dot(l0.T, l1_delta) # calculates how much to adjust each weight and adds them to the current weight.

print("Output After Training:")
print(l1)


