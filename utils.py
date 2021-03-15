import torch
import numpy as np
from scipy import integrate

def np_to_torch(npdata):
    '''
    transform numpy array to torch tensor
    :param npdata: a numpy array
    :return: a torch tensor
    '''
    data = npdata.astype('float')
    data = torch.from_numpy(data)
    # torch_data = torch.tensor(data, dtype=torch.float32)
    torch_data = data.clone().detach().float()
    return torch_data

def AILoss(x, ypred, ytrue, alpha=1):
    '''
    Calculate the area integral loss.
    :param x: x label - a numpy array
    :param ypred: predicted y value - a numpy array with the same shape as x
    :param ytrue: the true value - a float value
    :param alpha: the area attenuation factor
    :return: a float value
    '''
    def function(x1, x2, y1, y2):
        k = 0
        b = y1
        f = lambda t : 10 * np.exp(alpha*(t-ytrue)) * np.abs(k*t+b) / ytrue
        return f

    if x.shape[0] <= 1:
        return np.inf

    total_loss = 0
    for i in range(1, x.shape[0]):
        x1, x2 = x[i-1], x[i]
        y1, y2 = ypred[i-1] - ytrue, ypred[i] - ytrue

        if x1 > x2:
            return np.inf
        if x1 == x2:
            continue
        total_loss += integrate.quad(function(x1, x2, y1, y2), x1, x2)[0]
    return total_loss