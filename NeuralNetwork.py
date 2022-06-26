from telnetlib import XDISPLOC
from tkinter import N
import numpy as np
from Activation_ReLu import Activation_ReLu
from Activation_Softmax import Activation_Softmax
from Categorical_CrossEntropy import Categorical_CrossEntropy
from Layer import Layer



class NeuralNetworks:

    def __init__(self):
        self.layers = []
        self.activations = []


    def addLayer(self, n_inputs,n_neurons, activation):
        self.layers.append(Layer(n_inputs,n_neurons))

        if activation == 'Softmax':
            self.activations.append(Activation_Softmax())
        elif activation == 'ReLu':
            self.activations.append(Activation_ReLu())


    def forward(self,X):
        result = X

        for layer,activation in zip(self.layers,self.activations):
            layer.forward(result)
            activation.forward(layer.output)
            result = activation.output

        return result


    def backpropagation(self,X,y):
        prediction = self.forward(X)

        loss = Categorical_CrossEntropy()
        
        deltas = [None] * len(self.layers)

        deltas[-1] = np.mean(self.activations[-1].derivative() * loss.derivative(prediction,y),axis=0)


        for i in range(len(deltas)-2,-1,-1):
            deltas[i] = np.mean(np.dot(self.layers[i+1].weights,deltas[i+1]) * self.activations[i].derivative(),axis=0)
        

        outputs = [X] + [l.output for l in self.activations[:-1]]

        w_grad = [np.dot(deltas[i],outputs[i]) for i in range(len(outputs))]


        #Gradient Descent




if __name__ == "__main__":
    NN = NeuralNetworks()

    NN.addLayer(3,4,"ReLu")
    
    for _ in range(2):
        NN.addLayer(4,4,"ReLu")

    NN.addLayer(4,3,"Softmax")

    X = np.random.randn(3,3)
    y = [[0,0,1],[1,0,1],[0,1,0]]


    NN.backpropagation(X,y)
    