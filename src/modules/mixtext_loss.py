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

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        Lx = - \
            torch.mean(torch.sum(F.log_softmax(
                outputs_x, dim=1) * targets_x, dim=1))

        probs_u = torch.softmax(outputs_u, dim=1)

        Lu = F.kl_div(probs_u.log(), targets_u, None, None, 'batchmean')

        return Lx, Lu, self.lambda_u * self.linear_rampup(epoch)

    def linear_rampup(self, current):
        if self.rampup == 0:
            return 1.0
        else:
            current = np.clip(current / self.rampup, 0.0, 1.0)
            return float(current)
