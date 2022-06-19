from turtle import forward
import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    def calculate(self, output,y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    

    @abstractmethod
    def forward(self, y_pred, y_true):
        pass
    