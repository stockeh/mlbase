import torch
import torch.nn.functional as F


class DiceLoss(torch.nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    # based on:
    # https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super().__init__()
        self.eps: float = 1e-6

    def _onehot(self, labels: torch.Tensor, num_classes: int,
                device=None, dtype=None) -> torch.Tensor:
        shape = labels.shape
        one_hot = torch.zeros((shape[0], num_classes) +
                              shape[1:], device=device, dtype=dtype)
        return one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    def forward(self, Y: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Y: BxCxHxW, T: BxHxW"""
        # compute softmax over the classes axis
        Ysoft = F.softmax(Y, dim=1)

        # create the labels one hot tensor
        T1 = self._onehot(
            T, num_classes=Y.shape[1], device=Y.device, dtype=Y.dtype)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(Ysoft * T1, dims)
        cardinality = torch.sum(Ysoft + T1, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)
