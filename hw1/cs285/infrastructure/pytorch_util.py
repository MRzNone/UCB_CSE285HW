from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ac):
        super(ResBlock, self).__init__()

        self.net = nn.Linear(in_dim, out_dim)
        self.ac = ac

    def forward(self, x):
        return self.ac(self.net(x)) + x


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    # return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    layers_size = [input_size, *([size] * (n_layers + 1)), output_size]
    layers = []

    hidden_act = _str_to_activation[activation] if isinstance(activation, str) else activation
    output_act = _str_to_activation[output_activation] if isinstance(activation, str) else output_activation

    for i in range(len(layers_size) - 1):
        in_dim = layers_size[i]
        out_dim = layers_size[i + 1]


        if i == len(layers_size) - 2:
            ac = output_act
        else:
            ac = hidden_act

        if in_dim == out_dim:
            layers.append(ResBlock(in_dim, out_dim, ac))
        else:
            layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                ac,
            ))

    module = nn.Sequential(*layers)

    return module


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
