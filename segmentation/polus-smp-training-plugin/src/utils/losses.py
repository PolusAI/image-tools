import numpy as np
import torch
from torch.nn.modules.loss import _Loss


class MCCLoss(_Loss):
    
    def __init__(
        self,
    ):       
        super(MCCLoss, self).__init__()
  
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        eps = 1e-5
        bs = y_true.size(0)
        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        tp = torch.sum(torch.mul(y_pred, y_true)) + eps
        tn = torch.sum(torch.mul((1 - y_pred), (1 - y_true))) + eps
        fp = torch.sum(torch.mul(y_pred, (1 - y_true))) + eps
        fn = torch.sum(torch.mul((1 - y_pred), y_true)) + eps

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, 1, fp)
            * torch.add(tp, 1, fn)
            * torch.add(tn, 1, fp)
            * torch.add(tn, 1, fn)
        )
        # Adding 1 to the denominator to avoid divide-by-zero errors.
        mcc = torch.div(numerator.sum(), denominator.sum())
        loss = 1.0 - mcc

        return loss