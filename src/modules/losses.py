"""
Although NegEntropy is not a loss per se, it made sense to put them both in the same file for convenience's sake.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from torch.nn import CrossEntropyLoss, MSELoss
import torch
import torch.nn.functional as F
import numpy as np

class SemiLoss(object):

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, self.linear_rampup(epoch,warm_up)
    
    def linear_rampup(self, current, warm_up, lambda_u, rampup_length=16):
        current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
        return lambda_u*float(current)
    
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))