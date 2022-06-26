import numpy as np


class Activation_ReLu:
    def forward(self, inputs):
        f = lambda x: (x > 0) * x
        self.output = f(inputs)


    def derivative(self,inputs):
        f =  lambda x : (x > 0) * 1
        return f(inputs)

