import importlib
import mlx.nn as nn
import mlx.core as mx

from mlx.utils import tree_flatten, tree_map
from typing import List, Type, Tuple, Union, Optional


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def get_activation(activation_f: str) -> Type:
    package_name = 'mlx.nn.layers.activations'
    module = importlib.import_module(package_name)

    activations = [getattr(module, attr) for attr in dir(module)]
    activations = [cls for cls in activations if isinstance(
        cls, type) and issubclass(cls, nn.Module)]
    names = [cls.__name__.lower() for cls in activations]

    try:
        index = names.index(activation_f.lower())
        return activations[index]
    except ValueError:
        raise NotImplementedError(
            f'get_activation: {activation_f=} is not yet implemented.')


def compute_padding(input_size: tuple, kernel_size: int | tuple, stride: int | tuple = 1, dilation: int | tuple = 1):
    if len(input_size) == 2:
        input_size = (*input_size, 1)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    input_h, input_w, _ = input_size
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    # Compute the effective kernel size after dilation
    effective_kernel_h = (kernel_h - 1) * dilation_h + 1
    effective_kernel_w = (kernel_w - 1) * dilation_w + 1

    # Compute the padding needed for same convolution
    pad_h = max((input_h - 1) * stride_h + effective_kernel_h - input_h, 0)
    pad_w = max((input_w - 1) * stride_w + effective_kernel_w - input_w, 0)

    # Compute the padding for each side
    pad_top = pad_h // 2
    pad_left = pad_w // 2

    return (pad_top.item(), pad_left.item())


class Base(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    @property
    def num_params(self):
        return sum(x.size for k, x in tree_flatten(self.parameters()))

    @property
    def shapes(self):
        return tree_map(lambda x: x.shape, self.parameters())

    def summary(self):
        print(self)
        print(f'Number of parameters: {self.num_params}')

    def __call__(self, x: mx.array) -> mx.array:
        raise NotImplementedError('Subclass must implement this method')


class MLP(Base):
    def __init__(self, n_inputs: int, n_hiddens_list: Union[List, int],
                 n_outputs: int, activation_f: str = 'tanh'):
        super().__init__()

        if isinstance(n_hiddens_list, int):
            n_hiddens_list = [n_hiddens_list]

        if n_hiddens_list == [] or n_hiddens_list == [0]:
            self.n_hidden_layers = 0
        else:
            self.n_hidden_layers = len(n_hiddens_list)

        activation = get_activation(activation_f)

        self.layers = []
        ni = n_inputs
        if self.n_hidden_layers > 0:
            for _, n_units in enumerate(n_hiddens_list):
                self.layers.append(nn.Linear(ni, n_units))
                self.layers.append(activation())
                ni = n_units
        self.layers.append(nn.Linear(ni, n_outputs))

    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        for l in self.layers:
            x = l(x)
        return x


class Network(Base):
    def __init__(self,
                 n_inputs: Union[List[int], Tuple[int], mx.array],
                 n_outputs: int,
                 conv_layers_list: Optional[List[dict]] = None,
                 n_hiddens_list: Optional[Union[List, int]] = 0,
                 activation_f: str = 'relu'
                 ):
        super().__init__()

        if isinstance(n_hiddens_list, int):
            n_hiddens_list = [n_hiddens_list]

        if n_hiddens_list == [] or n_hiddens_list == [0]:
            self.n_hidden_layers = 0
        else:
            self.n_hidden_layers = len(n_hiddens_list)

        activation = get_activation(activation_f)

        ni = mx.array(n_inputs)
        self.conv = []
        if conv_layers_list:
            for conv_layer in conv_layers_list:
                n_channels = ni[-1].item()

                padding = conv_layer.get('padding', compute_padding(  # same padding
                    ni, conv_layer['kernel_size'], conv_layer.get('stride', 1), conv_layer.get('dilation', 1)))

                self.conv.extend([
                    nn.Conv2d(n_channels, conv_layer['filters'], conv_layer['kernel_size'],
                              stride=conv_layer.get('stride', 1), padding=padding,
                              dilation=conv_layer.get('dilation', 1), bias=conv_layer.get('bias', True)),
                    activation(),
                    nn.MaxPool2d(2, stride=2)
                ])
                ni = mx.concatenate(
                    [ni[:-1] // 2, mx.array([conv_layer['filters']])]
                )

        ni = mx.prod(ni).item()
        self.fcn = []
        if self.n_hidden_layers > 0:
            for _, n_units in enumerate(n_hiddens_list):
                self.fcn.append(nn.Linear(ni, n_units))
                self.fcn.append(activation())
                ni = n_units
        self.output = nn.Linear(ni, n_outputs)

    def __call__(self, x):
        for l in self.conv:
            x = l(x)
        x = x.reshape(x.shape[0], -1)
        for l in self.fcn:
            x = l(x)
        return self.output(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        x = self.norm(x)
        return self.w2(self.dropout(nn.silu(self.w1(x))) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.wq = nn.Linear(dim, inner_dim, bias=False)
        self.wk = nn.Linear(dim, inner_dim, bias=False)
        self.wv = nn.Linear(dim, inner_dim, bias=False)
        self.wo = nn.Linear(inner_dim, dim, bias=False)
        self.rope = nn.RoPE(dim_head, traditional=True, base=1e4)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def __call__(self, x, mask=None):
        b, n, d = x.shape
        x = self.norm(x)

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        reshaper = (lambda x: x.reshape(
            b, n, self.heads, -1).transpose(0, 2, 1, 3))
        queries, keys, values = map(reshaper, (queries, keys, values))

        queries = self.rope(queries)
        keys = self.rope(keys)

        # scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        # if mask is not None:
        #     scores += mask
        # scores = mx.softmax(scores.astype(mx.float32),
        #                     axis=-1).astype(scores.dtype)
        # output = (scores @ values)
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(b, n, -1)
        return self.wo(output)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = []
        for _ in range(depth):
            self.layers.append((
                Attention(dim, heads=heads,
                          dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ))

    def __call__(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)

        return x


class ViT(Base):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * \
            (image_width // patch_width)

        assert pool in {'cls', 'mean'}, \
            'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Conv2d(
            channels, dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embedding = mx.zeros((1, num_patches + 1, dim))
        self.cls_token = mx.zeros((1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.norm = nn.RMSNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes, bias=False)

    def __call__(self, x):
        x = self.to_patch_embedding(x)
        b, h, w, c = x.shape
        x = x.reshape(b, -1, c)

        cls_tokens = mx.repeat(self.cls_token, b, 0)
        x = mx.concatenate([cls_tokens, x], axis=1)

        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = mx.mean(x, axis=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(self.norm(x))
