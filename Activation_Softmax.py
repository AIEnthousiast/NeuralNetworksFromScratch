import numpy as np


class Activation_Softmax:

    def forward(self, inputs):
        expo_inputs = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        self.output = expo_inputs / np.sum(expo_inputs,axis=1,keepdims=True)
        
