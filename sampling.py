import torch
from torch_util import device, Module, random_indices


class SampleBuffer(Module):
    COMPONENT_NAMES = ('states', 'actions', 'next_states', 'rewards', 'dones')

    def __init__(self, state_dim, action_dim, capacity, discrete_actions=False,
                 device=device):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.discrete_actions = discrete_actions
        self.device = device

        self._bufs = {}
        self.register_buffer('_pointer', torch.tensor(0, dtype=torch.long))

        if discrete_actions:
            assert action_dim == 1
            action_dtype = torch.int
            action_shape = []
        else:
            action_dtype = torch.float
            action_shape = [action_dim]

        components = (
            ('states', torch.float, [state_dim]),
            ('actions', action_dtype, action_shape),
            ('next_states', torch.float, [state_dim]),
            ('rewards', torch.float, []),
            ('dones', torch.bool, [])
        )
        for name, dtype, shape in components:
            self._create_buffer(name, dtype, shape)

    def __len__(self):
        return min(self._pointer, self.capacity)

    @classmethod
    def from_state_dict(cls, state_dict, device=device):
        # Must have same keys
        assert set(state_dict.keys()) == {*(f'_{name}' for name in cls.COMPONENT_NAMES), '_pointer'}
        states, actions = state_dict['_states'], state_dict['_actions']

        # Check that length (size of first dimension) matches
        l = len(states)
        for name in cls.COMPONENT_NAMES:
            tensor = state_dict[f'_{name}']
            assert torch.is_tensor(tensor)
            assert len(tensor) == l

        # Capacity, dimensions, and type of action inferred from state_dict
        buffer = cls(state_dim=states.shape[1], action_dim=actions.shape[1], capacity=l,
                     discrete_actions=(not actions.dtype.is_floating_point),
                     device=device)
        buffer.load_state_dict(state_dict)
        return buffer

    def _create_buffer(self, name, dtype, shape):
        assert name not in self._bufs
        _name = f'_{name}'
        buffer_shape = [self.capacity, *shape]
        buffer = torch.empty(*buffer_shape, dtype=dtype, device=self.device)
        self.register_buffer(_name, buffer)
        self._bufs[name] = buffer

    def _get1(self, name):
        buf = self._bufs[name]
        if self._pointer <= self.capacity:
            return buf[:self._pointer]
        else:
            i = self._pointer % self.capacity
            return torch.cat([buf[i:], buf[:i]])

    def get(self, *names, device=device, as_dict=False):
        """
        Retrieves data from the buffer. Pass a vararg list of names.
        What is returned depends on how many names are given:
            * a list of all components if no names are given
            * a single component if one name is given
            * a list with one component for each name otherwise
        """
        if len(names) == 0:
            names = self.COMPONENT_NAMES
        bufs = [self._get1(name).to(device) for name in names]
        if as_dict:
            return dict(zip(names, bufs))
        else:
            return bufs if len(bufs) > 1 else bufs[0]

    def append(self, **kwargs):
        assert set(kwargs.keys()) == set(self.COMPONENT_NAMES)
        i = self._pointer % self.capacity
        for name in self.COMPONENT_NAMES:
            self._bufs[name][i] = kwargs[name]
        self._pointer += 1

    def extend(self, **kwargs):
        assert set(kwargs.keys()) == set(self.COMPONENT_NAMES)
        batch_size = len(list(kwargs.values())[0])
        assert batch_size <= self.capacity, 'We do not support extending by more than buffer capacity'
        i = self._pointer % self.capacity
        end = i + batch_size
        if end <= self.capacity:
            for name in self.COMPONENT_NAMES:
                self._bufs[name][i:end] = kwargs[name]
        else:
            fit = self.capacity - i
            overflow = end - self.capacity
            # Note: fit + overflow = batch_size
            for name in self.COMPONENT_NAMES:
                buf, arg = self._bufs[name], kwargs[name]
                buf[-fit:] = arg[:fit]
                buf[:overflow] = arg[-overflow:]
        self._pointer += batch_size

    def sample(self, batch_size, replace=True, device=device, include_indices=False):
        indices = torch.randint(len(self), [batch_size], device=device) if replace else \
            random_indices(len(self), size=batch_size, replace=False)
        bufs = [self._bufs[name][indices].to(device) for name in self.COMPONENT_NAMES]
        return (bufs, indices) if include_indices else bufs

