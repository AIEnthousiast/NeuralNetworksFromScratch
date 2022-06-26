# NeuralNetworksFromScratch
A cool little project. My main aim is to code a neural network from the ground-up. At the end, it will be tested on the MNSIT dataset. 


# What is a neural network ?


A neural network (NN) is a set of artificial neurons displayed in layers for the purpose of learning a pattern. It contains a set of inputs , a set of outputs and intermediary layers of neurons. Each of these neurons is characterized by  a set of weights, one bias and activation function. 

# What happens inside of a neuron ?

A neuron receives the inputs formed by the values inside all the neurons of the previous layers (feed forward). Then , it computes with its set of weights a linear combination of these inputs , add to it its bias and then evaluate the result by its activation function. 

# What is an activation function and why is it necessary ?

An activation function is a non-linear function given to a neuron for the sake of computing its value. It is necessary if the NN wants to learn a non linear problem (thing about leaning the curve of x --> x^2 for example : no line can approximate it truthfully). 
We have for example: sigmoid, ReLu , softmax...

# How can a network lean? 

Learning for a NN is reduced to altering the values its weights and biases. In order to do so, we use an algorithm called backpropagation.    


# TODO BACKPROPAGATION