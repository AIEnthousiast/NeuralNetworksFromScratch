from codecs import backslashreplace_errors
from curses.ascii import SI
from turtle import back, dot
import numpy as np
from Layer import Layer
from Activation_Softmax import Activation_Softmax
from Activation_ReLu import Activation_ReLu
from Categorical_CrossEntropy import Categorical_CrossEntropy



class SingleInputNeuralNetwork:

    def __init__(self):
        self.layers = []
        self.activations = []
    
    def addALayer(self, n_inputs, n_neurons,activation):
        self.layers.append(Layer(n_inputs,n_neurons))
        self._addActivation(activation)

    def _addActivation(self, activation):
        if activation == "Softmax":
            self.activations.append(Activation_Softmax())
        
        elif activation == "ReLu":
            self.activations.append(Activation_ReLu())




    def gradient_descent(self,X,y,r):
        delta_w, delta_b = self.backpropagation(X, y)


        for i in range(len(self.layers)):
            #print("Avant:",self.layers[i].weights)
            self.layers[i].weights -= delta_w[i]*r
            #print("Après:",self.layers[i].weights)
            
            
            self.layers[i].biases -= delta_b[i].reshape(1,-1)*r


    
    def forward(self, X):

        result = X 

        for layer,activation in zip(self.layers,self.activations):
            layer.forward(result)
            activation.forward(layer.output)
            result = activation.output

        return result
    

    def backpropagation(self,X,y):
        deltas = []

        loss = Categorical_CrossEntropy()


        loss_gradient = loss.gradient(self.activations[-1].output,y)

        deltas.append(loss_gradient * self.activations[-1].derivative(self.layers[-1].output))

        for i in range(len(self.layers )-2,-1,-1):

            dot_prod = np.dot(deltas[-1],self.layers[i+1].weights.T) 
            d = dot_prod * self.activations[i].derivative(self.layers[i].output)
            deltas.append(d)


        deltas = deltas[::-1]
    
        delta_b = deltas
        delta_w = []

        activations = [X] + [a.output for a in self.activations[:-1]]


        for d,activation in zip(deltas,activations):
            delta_w.append(np.dot(activation.reshape(-1,1),d))

        return delta_w, delta_b


trainingX = np.random.randn(3,5)
target = np.array([[1, 0, 0, 0],
                [ 0, 1 , 0, 0],
                [0,1,0,0]])

print(target.shape)

sIN  = SingleInputNeuralNetwork()

sIN.addALayer(5,4, "ReLu")
sIN.addALayer(4,5, "ReLu")
sIN.addALayer(5,5, "ReLu")
sIN.addALayer(5,4,"Softmax")


 

loss = Categorical_CrossEntropy()


for _ in range(10000):
    result = sIN.forward(trainingX)
    sIN.backpropagation(trainingX,target)

    sIN.gradient_descent(trainingX,target, 0.01)

    print(loss.calculate(result,target))
 
