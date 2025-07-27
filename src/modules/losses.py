"""
Although NegEntropy is not a loss per se, it made sense to put them both in the same file for convenience's sake.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn.functional as F
import numpy as np

class SemiLoss(object):

    def __init__(self, lambda_u, rampup):
        self.lambda_u = lambda_u
        self.rampup = rampup

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, self.linear_rampup(epoch, warm_up)
    
    def linear_rampup(self, current, warm_up):
        current = np.clip((current-warm_up) / self.rampup, 0.0, 1.0)
        return self.lambda_u*float(current)
    
class NegEntropy(object):
    """
    This loss is used to penalise the model for being too confident in its predictions.
    Useful for asymmetric label noise, where a label is assigned to a sample in a systematic way.
    This aligns with our task as we systematically assign the same label to an entire subreddits' worth of samples.
    """
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))