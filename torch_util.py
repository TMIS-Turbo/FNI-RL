import numpy as np
import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def torchify(x, double_to_float=True, int_to_long=True, to_device=True):
    if torch.is_tensor(x):
        pass
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        x = torch.tensor(x)

    if x.dtype == torch.double:
        if double_to_float:
            x = x.float()
    elif x.dtype == torch.int:
        if int_to_long:
            x = x.long()

    if to_device:
        x = x.to(device)

    return x


def numpyify(x):
    if isinstance(x, np.ndarray):
        return x
    elif torch.is_tensor(x):
        return x.cpu().numpy()
    else:
        return np.array(x)


def random_indices(high, size=None, replace=False, p=None):
    assert isinstance(high, int)
    p_np = numpyify(p) if torch.is_tensor(p) else p
    return torchify(np.random.choice(high, size=size, replace=replace, p=p_np))

def random_choice(tensor, size=None, replace=False, p=None, dim=0):
    indices = random_indices(tensor.shape[dim], size=size, replace=replace, p=p)
    return tensor.index_select(dim, indices)

def quantile(a, q, dim=None, as_torch=True, to_device=False):
    quantile = np.quantile(numpyify(a), numpyify(q), axis=dim)
    if as_torch:
        return torchify(quantile, to_device=to_device)
    elif quantile.ndim == 0:
        return float(quantile)
    else:
        return list(map(float, quantile))


class Module(nn.Module):
    def __call__(self, *args, **kwargs):
        args = [x.to(device) if isinstance(x, torch.Tensor) else x for x in args]
        kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return super().__call__(*args, **kwargs)

    def save(self, f, prefix='', keep_vars=False):
        state_dict = self.state_dict(prefix=prefix, keep_vars=keep_vars)
        torch.save(state_dict, f)

    def load(self, f, map_location=device, strict=True):
        state_dict = torch.load(f, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)


class DummyModuleWrapper:
    """Use this class if you want to exclude a submodule from being included in its parent's state_dict.
    This is useful for modules which are static and/or use a lot of memory, such as sample buffers."""
    def __init__(self, module):
        assert isinstance(module, nn.Module)
        self.__dict__['_module'] = module

    def __getattr__(self, attr):
        if attr == '_module':
            return self.__dict__['_module']
        else:
            return getattr(self._module, attr)

    def __setattr__(self, attr, value):
        setattr(self.__dict__['_module'], attr, value)

    def __len__(self):
        return len(self._module)


class Squeeze(nn.Module):
    """A layer that simply squeeze its input"""
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


default_init_w = nn.init.xavier_normal_
default_init_b = nn.init.zeros_


def weight_initializer(init_w=default_init_w, init_b=default_init_b):
    def init_fn(m):
        if hasattr(m, 'weight'):
            init_w(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_b(m.bias)
    return init_fn


def dry_run(module, input_dim):
    """Just runs the network forward once and ignores errors.
    Fixes an uninformative PyTorch/CUDA error I was having, but not sure why."""
    try:
        with torch.no_grad():
            module(torchify(np.zeros((1, input_dim))))
    except:
        pass


# Modify the dict below to add activation functions
KNOWN_ACTIVATIONS = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'softmax': lambda: nn.Softmax(dim=1),
    'softplus': nn.Softplus,
    'tanh': nn.Tanh
}


def _process_activation(activation):
    if isinstance(activation, str):
        assert activation in KNOWN_ACTIVATIONS, f'Unknown activation: {activation}'
        get_activation = KNOWN_ACTIVATIONS[activation]
    else:
        get_activation = activation
    assert callable(get_activation)
    return get_activation


def mlp(dims, layer_factory=nn.Linear, activation='relu', output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'
    get_activation = _process_activation(activation)

    layers = []
    for i in range(n_dims - 2):
        layers.append(layer_factory(dims[i], dims[i+1]))
        layers.append(get_activation())
    layers.append(layer_factory(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(_process_activation(output_activation)())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(1))
    net = nn.Sequential(*layers)
    net.apply(weight_initializer())
    net.to(device=device, dtype=torch.float)
    dry_run(net, dims[0])
    return net


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def update_ema(target, source, rate):
    assert 0 <= rate <= 1
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(rate * param.data + (1 - rate) * target_param.data)
