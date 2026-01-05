import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Focal Loss for Multi-Label Classification.

        Args:
            alpha (float): Weighting factor for the rare class (positive examples).
            gamma (float): Focusing parameter. Higher values focus more on hard examples.
            reduction (str): 'mean' or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # BCEWithLogitsLoss combines Sigmoid and BCE for numerical stability
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # pt is the probability of the true class being predicted
        # pt = exp(-bce_loss)
        pt = torch.exp(-bce_loss)

        # Focal term: (1 - pt)^gamma
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
 