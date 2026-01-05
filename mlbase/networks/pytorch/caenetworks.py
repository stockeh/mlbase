import numpy as np
import torch
from torch.nn import functional as F

from mlbase.networks.pytorch.neuralnetworks import NeuralNetwork


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
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def _onehot(
        self, labels: torch.Tensor, num_classes: int, device=None, dtype=None
    ) -> torch.Tensor:
        shape = labels.shape
        one_hot = torch.zeros(
            (shape[0], num_classes) + shape[1:], device=device, dtype=dtype
        )
        return one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    def forward(self, Y: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Y: BxCxHxW, T: BxHxW"""
        # compute softmax over the classes axis
        Ysoft = F.softmax(Y, dim=1)

        # create the labels one hot tensor
        T1 = self._onehot(T, num_classes=Y.shape[1], device=Y.device, dtype=Y.dtype)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(Ysoft * T1, dims)
        cardinality = torch.sum(Ysoft + T1, dims)

        dice_score = 2.0 * intersection / (cardinality + self.eps)
        return torch.mean(1.0 - dice_score)


class ConvolutionalAutoEncoder(NeuralNetwork):

    class Network(torch.nn.Module):
        def __init__(
            self, n_inputs, n_hiddens_list, n_outputs, conv_layers, activation_f
        ):
            super().__init__()
            if not isinstance(n_hiddens_list, list):
                raise Exception("Network: n_hiddens_list must be a list.")

            assert conv_layers is not None, "Must include Conv Layers."
            assert n_inputs == n_outputs, "n inputs must equal n outputs."

            if len(n_hiddens_list) == 0 or n_hiddens_list[0] == 0:
                self.n_hidden_layers = 0
            else:
                self.n_hidden_layers = len(n_hiddens_list)

            # network varaibles
            self.n_inputs = n_inputs
            self.n_hiddens_list = n_hiddens_list
            self.n_outputs = n_outputs
            self.conv_layers = conv_layers

            # build nnet
            self.model = torch.nn.ModuleList()

            activations = [
                torch.nn.Tanh,
                torch.nn.ReLU,
                torch.nn.Sigmoid,
            ]
            names = [str(o.__name__).lower() for o in activations]
            try:
                activation = activations[names.index(str(activation_f).lower())]
            except:
                raise NotImplementedError(
                    f"__init__: {activation_f=} is not yet implemented."
                )
            l = 0
            ni = np.asarray(n_inputs)
            # add convolutional layers
            # TODO: currently only works with padding='same' and stride=1
            for conv_layer in self.conv_layers:
                n_channels = ni[0]  # C, H, W
                self.model.add_module(
                    f"conv_{l}",
                    torch.nn.Conv2d(
                        n_channels,
                        conv_layer["n_units"],
                        conv_layer["shape"],
                        stride=1,
                        padding="same",
                        padding_mode="zeros",
                    ),
                )
                self.model.add_module(f"activation_{l}", activation())
                self.model.add_module(f"maxpool_{l}", torch.nn.MaxPool2d(2, stride=2))
                # TODO: currently only to divide H, W dimensions by 2
                # with 'same' padding
                ni = np.concatenate([[conv_layer["n_units"]], ni[1:] // 2])
                l += 1

            conv_shape = ni
            self.model.add_module("flatten", torch.nn.Flatten())  # okay
            ni = np.prod(ni)

            # fully-connected latent layers
            # TODO: only Tanh activations are supported
            if self.n_hidden_layers > 0:
                for _, n_units in enumerate(n_hiddens_list):
                    self.model.add_module(f"linear_{l}", torch.nn.Linear(ni, n_units))
                    self.model.add_module(f"activation_{l}", torch.nn.Tanh())
                    ni = n_units
                    l += 1
            self.model.add_module(
                f"linear_{l}", torch.nn.Linear(ni, np.prod(conv_shape))
            )
            self.model.add_module(f"activation_{l}", torch.nn.Tanh())
            l += 1

            # decoder
            self.model.add_module(
                f"unflatten",
                torch.nn.Unflatten(
                    1, unflattened_size=tuple([int(i) for i in conv_shape])
                ),
            )  # hack to force tuple of ints
            ni = conv_shape
            for conv_layer in reversed(self.conv_layers):
                n_channels = ni[0]  # C, H, W
                self.model.add_module(
                    f"conv_{l}",
                    torch.nn.Conv2d(
                        n_channels,
                        conv_layer["n_units"],
                        conv_layer["shape"],
                        stride=1,
                        padding="same",
                        padding_mode="zeros",
                    ),
                )
                self.model.add_module(f"activation_{l}", activation())
                self.model.add_module(
                    f"upsample_{l}",
                    torch.nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=True
                    ),
                )
                # TODO: currently only to divide H, W dimensions by 2
                # with 'same' padding
                ni = np.concatenate([[conv_layer["n_units"]], ni[1:] * 2])
                l += 1

            # Linear Convolutional output layer
            self.model.add_module(
                f"output_{l}",
                torch.nn.Conv2d(
                    ni[0],
                    n_outputs[0],
                    self.conv_layers[0]["shape"],
                    stride=1,
                    padding="same",
                    padding_mode="zeros",
                ),
            )

            # self.model.apply(self._init_weights)

        def _init_weights(self, m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        def forward_all_outputs(self, X):
            Ys = [X]
            for i, layer in enumerate(self.model):
                Ys.append(layer(Ys[i]))
            return Ys[1:]  # all outputs without original inputs

        def forward(self, X):
            Ys = self.forward_all_outputs(X)
            return Ys[-1]

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
