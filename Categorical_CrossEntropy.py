import numpy as np
from Loss import Loss

class Categorical_CrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        clipped_y_pred= np.clip(y_pred, 1e-7, 1_1e-7)

        
        if len(y_true.shape) == 2:
            #One Hot Encoding
            values = np.sum(clipped_y_pred * y_true, axis=1)

        elif len(y_true.shape) == 1:
            values = y_pred[range(len(y_pred)),y_true]

        return -np.log(values)

    def gradient(self, pred, reel):
        return pred - reel
