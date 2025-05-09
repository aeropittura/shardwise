"""Module base classes and containers."""

import warnings
from collections import OrderedDict
import shardwise
import torch


class Module:
    """
    Base Module class that mimics the torch.nn.Module class.
    """

    # allow for versioning of modules:
    _version = 1
    SUPPORTS_PLAINTEXT_INPUTS = False

    def __init__(self):
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._modules = OrderedDict()
        self.encrypted = False
        self.train()

    def __repr__(self):
        encrypted_str = "encrypted" if self.encrypted else "unencrypted"
        return f"{type(self).__name__} {encrypted_str} module"

    @staticmethod
    def from_onnx(attributes=None):
        """
        Constructs a Shardwise module from an ONNX Protobuf string or file.
        """
        raise NotImplementedError("Call this function on a Module type.")

    def forward(self, *args, **kwargs):
        """Perform forward pass on model."""
        raise NotImplementedError("forward not implemented")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode=True):
        """Sets the module in the specified training mode."""
        for param in self.parameters():
            param.requires_grad = mode
        self.training = mode

        # Recursively set train mode
        for module in self.children():
            module.train(mode=mode)
        return self

    def eval(self):
        """Sets the module in evaluation mode."""
        return self.train(False)

    def children(self):
        """Returns an iterator over immediate children modules.
        Yields:
            Module: a child module
        """
        for _, module in self.named_children():
            yield module

    def named_children(self):
        """Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.
        Yields:
            (string, Module): Tuple containing a name and child module
        Example::
            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)
        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def register_module(self, name, module):
        """Registers child module in the module."""
        self._modules[name] = module

    def modules(self):
        """Returns an iterator over all modules in the network.
        Yields:
            Module: a module in the network
        Note:
            Duplicate modules are returned only once.
        """
        for _, module in self.named_modules():
            yield module

    def named_modules(self, memo=None, prefix=""):
        """Returns iterator over named modules (non-recursively)."""
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self.named_children():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def register_parameter(self, name, param, requires_grad=True):
        """
        Register parameter in the module. This function cannot register
        parameters in child modules.
        """
        if name in self._parameters or hasattr(self, name):
            raise ValueError("Parameter or field %s already exists." % name)
        param.requires_grad = requires_grad
        self._parameters[name] = param
        setattr(self, name, param)

    def set_parameter(self, name, param):
        """
        Sets value of parameter in the module. This function cannot set
        parameters in child modules.
        """
        if name not in self._parameters or not hasattr(self, name):
            raise ValueError("Parameter %s does not exist." % name)
        self._parameters[name] = param
        setattr(self, name, param)

    def set_parameter_from_shares(self, name, share, **kwargs):
        """
        Sets value of parameter in the module from shares. This functionality is
        only for MPC-encrypted models.

        Supported named arguments for `SecureTensor` parameters include the `precision`
        of the encoder (default = `None`), the rank of the `src` (default = 0),
        and the `ptype` of the shares (default = `shardwise.sharing.arithmetic`).

        This function cannot set the parameters in child modules.
        """

        # functionality is only supported when parameters are SecureTensors:
        assert self.encrypted, "can only set parameters from shares in encrypted models"
        if name not in self._parameters or not hasattr(self, name):
            raise ValueError("Parameter %s does not exist." % name)
        cls = type(self._parameters[name])
        assert hasattr(
            self._parameters[name], "from_shares"
        ), "parameter type {} does not supporting setting from shares".format(cls)

        # load parameters from shares:
        self._parameters[name] = cls.from_shares(share, **kwargs)
        setattr(self, name, self._parameters[name])

    def parameters(self, recurse=True):
        """Returns an iterator over module parameters.
        This is typically passed to an optimizer.
        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
        Yields:
            ShardTensor or torch.Tensor: module parameter
        """
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, recurse=True, prefix=None):
        """
        Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.
        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
        Yields:
            (string, ShardTensor or torch.Tensor): Tuple containing the name and parameter
        """
        for name, param in self._parameters.items():
            param_name = name if prefix is None else prefix + "." + name
            yield param_name, param
        if recurse:
            for module_name, module in self.named_children():
                pre = module_name if prefix is None else prefix + "." + module_name
                yield from module.named_parameters(recurse=recurse, prefix=pre)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """
        Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. The specified `prefix` will be
        used in names of parameters and buffers in this module. The `keep_vars`
        boolean determines if parameters and buffers are kept on the autograd tape.
        """
        for name, param in self.named_parameters(recurse=False):
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buffer in self.named_buffers(recurse=False):
            if buffer is not None:
                destination[prefix + name] = buffer if keep_vars else buffer.detach()

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Returns a dictionary containing the state of the module. Both parameters
        and persistent buffers (e.g., running averages) are included. Keys are
        corresponding parameter and buffer names.
        """

        # save parameters and buffers of current module:
        if destination is None:
            destination = OrderedDict()
            destination._metadata = {"version": Module._version}
        self._save_to_state_dict(destination, prefix, keep_vars)

        # recurse over modules:
        for name, module in self.named_children():
            if module is not None:
                module.state_dict(destination, prefix + name + ".", keep_vars=keep_vars)
        return destination

    def _load_from_state_dict_shardwise(self, state_dict, prefix, strict):
        """
        Copies parameters and buffers from `state_dict` into only this module
        but not its children. This is called on every submodule in the
        `load_state_dict` function.
        """

        # get state dict for just the current module (without children)
        local_state = {
            key: val for key, val in self.named_parameters() if val is not None
        }

        # in strict mode, check for missing keys in the state_dict:
        if strict:
            for name in local_state.keys():
                key = prefix + name
                if key not in state_dict:
                    raise ValueError("Key {} not found in state dict.".format(key))

        # loop over parameters / buffers in module:
        for name, param in local_state.items():
            key = prefix + name
            input_param = state_dict[key]

            # size in state_dict should match size of parameters:
            if input_param.size() != param.size():
                raise ValueError(
                    "Size mismatch for {}: copying a param with"
                    "shape {} from checkpoint, the shape in"
                    "current model is {}.".format(key, input_param.size(), param.size())
                )
                continue

            # cannot copy encrypted tensors into unencrypted models and vice versa:
            param_encrypted = isinstance(input_param, shardwise.ShardTensor)
            if param_encrypted:
                assert (
                    self.encrypted
                ), "cannot copy encrypted parameters into unencrypted model"
            else:
                assert (
                    not self.encrypted
                ), "cannot copy unencrypted parameters into encrypted model"

            # copy parameters from state_dict:
            with shardwise.no_grad(), torch.no_grad():
                param.copy_(input_param)

    def load_state_dict(self, state_dict, strict=True):
        """
        Copies parameters and buffers from `state_dict` into this module and its
        children. If `strict` is `True`, then the keys of `state_dict` must
        exactly match the keys returned by this module's `state_dict` function.
        """
        # check version of state_dict:
        if strict:
            metadata = getattr(state_dict, "_metadata", None)
            if metadata is None:
                raise ValueError("Specified state_dict does not have metadata.")
            version = metadata.get("version", -1)
            if version != Module._version:
                raise ValueError(
                    "Specified state_dict has incorrect version: {}".format(version)
                )

        # make copy state_dict so that load() can modify it:
        state_dict = state_dict.copy()

        def load(module, prefix=""):
            """
            Closure that performs the loading of a module recursively.
            """
            module._load_from_state_dict_shardwise(state_dict, prefix, strict)
            for name, child in module.named_children():
                if child is not None:
                    load(child, prefix + name + ".")

        # perform the actual loading:
        load(self)
        load = None  # break load->load reference cycle

    def zero_grad(self):
        """Sets gradients of all parameters to zero."""
        for param in self.parameters():
            param.grad = None

    def update_parameters(self, learning_rate, grad_threshold=100):
        """Performs gradient step on parameters.

        Parameters:
            grad_threshold - Because arithmetic operations can extremely rarely
                    return large incorrect results, we zero-out all elements
                    with magnitude larger than this given threshold. To turn
                    off thresholding, set to `None`.
        """
        assert self.training, "module not in training mode"
        with shardwise.no_grad(), torch.no_grad():
            for param in self.parameters():
                if param.grad is None:
                    continue
                if not self.encrypted and isinstance(param.grad, shardwise.ShardTensor):
                    raise RuntimeError(
                        "Cannot update parameters of unencrypted "
                        "model using encrypted gradients. Encrypt "
                        "model before updating parameters."
                    )

                # Threshold gradients to prevent gradient explosion from wrap overflow
                if self.encrypted and grad_threshold is not None:
                    # Compute based on square value since abs is more expensive
                    square_threshold = grad_threshold * grad_threshold
                    grad = param.grad.mul(param.grad.square().lt(square_threshold))
                else:
                    grad = param.grad

                param.sub_(grad.mul_(learning_rate))

    def register_buffer(self, name, buffer):
        """
        Register buffer in the module. Buffers are encrypted like parameters but
        they are not updated by parameter updates.
        """
        if name in self._buffers or hasattr(self, name):
            raise ValueError("Buffer or field %s already exists." % name)
        buffer.requires_grad = False
        self._buffers[name] = buffer
        setattr(self, name, buffer)

    def set_buffer(self, name, buffer):
        """
        Sets value of buffer in the module. This function cannot set the
        parameters in child modules.
        """
        if name not in self._buffers or not hasattr(self, name):
            raise ValueError("Buffer %s does not exist." % name)
        self._buffers[name] = buffer
        setattr(self, name, buffer)

    def buffers(self, recurse=True):
        """Returns an iterator over module buffers.
        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.
        Yields:
            ShardTensor or torch.Tensor: module buffer
        """
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer

    def named_buffers(self, recurse=True, prefix=None):
        """Returns an iterator over module buffers, yielding both the
        name of the buffer as well as the buffer itself.
        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.
        Yields:
            (string, ShardTensor or torch.Tensor): Tuple containing the name and buffer
        Example::
            >>> for name, buf in self.named_buffers():
            >>>    if name in ['running_var']:
            >>>        print(buf.size())
        """
        for name, buffer in self._buffers.items():
            buffer_name = name if prefix is None else prefix + "." + name
            yield buffer_name, buffer
        if recurse:
            for module_name, module in self.named_children():
                pre = module_name if prefix is None else prefix + "." + module_name
                yield from module.named_buffers(recurse=recurse, prefix=pre)

    def to(self, *args, **kwargs):
        """
        Moves and/or casts the parameters and buffers.

        This can be called as

        `to(device=None, dtype=None, non_blocking=False)`
        `to(dtype, non_blocking=False)`
        `to(tensor, non_blocking=False)`
        `to(memory_format=torch.channels_last)`

        Args:
            device (torch.device) – the desired device of the parameters
            and buffers in this module
            dtype (torch.dtype) – the desired floating point type of the
            floating point parameters and buffers in this module
            tensor (torch.Tensor) – Tensor whose dtype and device are the
            desired dtype and device for all parameters and buffers in this module
            memory_format (torch.memory_format) – the desired memory format
            for 4D parameters and buffers in this module (keyword only argument)

        """
        for name, param in self._parameters.items():
            self.set_parameter(name, param.to(*args, **kwargs))
        for name, buffer in self._buffers.items():
            self.set_buffer(name, buffer.to(*args, **kwargs))

        for module in self.children():
            module.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        """
        Moves all model parameters and buffers to the GPU.

        Args:
            device (int, optional) – if specified, all parameters will be copied
            to that device
        """
        for name, param in self._parameters.items():
            self.set_parameter(name, param.cuda(device=device))
        for name, buffer in self._buffers.items():
            self.set_buffer(name, buffer.cuda(device=device))

        for module in self.children():
            module.cuda(device=device)
        return self

    def cpu(self):
        """Moves all model parameters and buffers to the CPU."""
        for name, param in self._parameters.items():
            self.set_parameter(name, param.cpu())
        for name, buffer in self._buffers.items():
            self.set_buffer(name, buffer.cpu())

        for module in self.children():
            module.cpu()
        return self

    def _apply(self, fn):
        """Applies a function recursively on all modules."""
        for module in self.children():
            module._apply(fn)
        fn(self)
        return self

    def encrypt(self, mode=True, src=0):
        """Encrypts the model."""
        if mode != self.encrypted:
            # encrypt / decrypt parameters:
            self.encrypted = mode
            for name, param in self.named_parameters(recurse=False):
                requires_grad = param.requires_grad
                if mode:  # encrypt parameter
                    self.set_parameter(
                        name,
                        shardwise.tensor(
                            param, **{"src": src}, requires_grad=requires_grad
                        ),
                    )
                else:  # decrypt parameter
                    self.set_parameter(name, param.get_plain_text())
                    self._parameters[name].requires_grad = requires_grad

            # encrypt / decrypt buffers:
            for name, buffer in self.named_buffers(recurse=False):
                # encrypt buffer only if it's a torch tensor (not shapes)
                if mode and torch.is_tensor(buffer):
                    self.set_buffer(
                        name,
                        shardwise.tensor(buffer, **{"src": src}, requires_grad=False),
                    )
                # decrypt buffer if it's a shard_tensor
                elif isinstance(buffer, shardwise.ShardTensor):
                    self.set_buffer(name, buffer.get_plain_text())
                    self._buffers[name].requires_grad = False

            # apply encryption recursively:
            return self._apply(lambda m: m.encrypt(mode=mode, src=src))
        return self

    def decrypt(self):
        """Decrypts model."""
        return self.encrypt(mode=False)

    def __getattribute__(self, name):
        if name != "forward":
            return object.__getattribute__(self, name)

        def forward_function(*args, **kwargs):
            """
            Silently encrypted Torch inputs tensors (deprecated).
            """
            if self.encrypted and not self.SUPPORTS_PLAINTEXT_INPUTS:
                if any(torch.is_tensor(arg) for arg in args):
                    warnings.warn(
                        "Earlier versions of Shardwise silently encrypted Torch tensors. "
                        "That behavior is now deprecated because it is dangerous. "
                        "Please make sure you feed your model ShardTensors when needed.",
                        DeprecationWarning,
                    )
            elif not self.encrypted:
                if any(isinstance(arg, shardwise.ShardTensor) for arg in args):
                    raise RuntimeError(
                        "Cannot input ShardTensors into unencrypted model. "
                        "Encrypt the model before feeding it ShardTensors."
                    )
            return object.__getattribute__(self, name)(*tuple(args), **kwargs)

        return forward_function

    def __getattr__(self, name):
        """Redefine __getattr__ so that any parameters, modules or buffers
        inside the Module object can be accessed as attributes
        """
        if "_parameters" in self.__dict__:
            parameters = self.__dict__["_parameters"]
            if name in parameters:
                return parameters[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        if "_buffers" in self.__dict__:
            buffers = self.__dict__["_buffers"]
            if name in buffers:
                return buffers[name]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, name)
        )

    def __setattr__(self, name, value):
        """Redefine __setattr__ so that any submodules created
        inside the Module object are registered with _modules
        OrderedDict.
        """

        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        modules = self.__dict__.get("_modules")
        if isinstance(value, Module):
            if modules is None:
                raise AttributeError(
                    "cannot assign module before Module.__init__() call"
                )
            remove_from(self.__dict__, self._parameters, self._buffers)
            modules[name] = value
        elif modules is not None and name in modules:
            if value is not None:
                raise TypeError(
                    "cannot assign '{}' as child module '{}' "
                    "(torch.nn.Module or None expected)".format(
                        torch.typename(value), name
                    )
                )
            modules[name] = value
        else:
            for key in ["_parameters", "_modules", "_buffers"]:
                if key in self.__dict__ and name in self.__dict__[key]:
                    values = self.__dict__[key]
                    values[name] = value
                    return
            object.__setattr__(self, name, value)

    def add_module(self, name, module):
        """Adds and registers a submodule with a given name"""
        assert name not in self._modules.keys(), "Module %s already exists." % name
        self.register_module(name, module)


class Container(Module):
    """
    Container allows distinguishing between individual modules and containers.
    """

    pass


class Graph(Container):
    """
    Acyclic graph of modules.

    The module maintains a dict of named modules and a graph structure stored in
    a dict where each key is a module name, and the associated value is a list
    of module names that provide the input into the module. Each module may have
    an optional list of output names as well, that is stored as an attribute in
    the module by the `add_module` function.
    """

    def __init__(self, input_names, output_names, modules=None, graph=None):
        """
        Initializes a graph module with inputs named by `input_names`, that
        produces outputs named by `output_names`.

        Optionally, `modules` and the `graph` structure can be specified as well.
        Alternatively, the graph can be built up using the `add_module` function.
        """
        super().__init__()
        if not isinstance(input_names, (list, tuple)):
            input_names = [input_names]
        if not isinstance(output_names, (list, tuple)):
            output_names = [output_names]
        self.input_names = input_names
        self.output_names = output_names
        self._graph = {}
        if modules is not None:
            self._modules = modules
        if graph is not None:
            self._graph = graph

    def add_module(self, name, module, input_names=None, output_names=None):
        """
        Adds a `module` with the specified `name` to the graph. If the `module`
        expects inputs, their names should be specified via `input_names`.

        The module is expected to produce an output with `name`. However, if the
        module produces multiple outputs, these must be named in the
        `output_names` list.

        Both `input_names` and `output_names` are expected to be ordered.
        """
        assert name not in self._graph, "Module %s already exists." % name
        self.register_module(name, module)
        if input_names is not None:
            self._graph[name] = input_names
        if output_names is not None:
            module._output_names = output_names

    def forward(self, *args):
        assert len(args) == len(
            self.input_names
        ), f"Expected {len(self.input_names)} inputs but received {len(args)}."

        # keep track of all values that have been computed:
        values = {self.input_names[idx]: args[idx] for idx in range(len(args))}
        computed = {key: False for key in self._graph.keys()}
        inputs_available = {
            key: [False for _ in range(len(value_list))]
            for key, value_list in self._graph.items()
        }

        def _mark_as_computed(name):
            """Marks a value as having been computed."""
            computed[name] = True
            for key, value_list in self._graph.items():
                if name in value_list:
                    inputs_available[key][value_list.index(name)] = True

        def _find_computable_node():
            """Find a node for which all inputs are available."""
            for key, inputs_available_list in inputs_available.items():
                if all(inputs_available_list) and not computed[key]:
                    return key
            return None

        def _clear_unused_values():
            """Clear values that are no longer needed (to save memory)."""
            remove_keys = []
            for remove_key in values.keys():
                can_be_removed = True

                # we cannot remove a value if it is still needed:
                for key, value_list in self._graph.items():
                    if not computed[key] and remove_key in value_list:
                        can_be_removed = False
                        break
                if can_be_removed:
                    remove_keys.append(remove_key)

            # remove all values we no longer need:
            for remove_key in remove_keys:
                del values[remove_key]
            # NOTE: We maintain inputs_available[remove_key] as True to
            # prevent re-computation of the node.

        # perform forward pass:
        for input_name in self.input_names:
            _mark_as_computed(input_name)
        node_to_compute = _find_computable_node()
        while node_to_compute is not None:
            # compute output of module:
            input = [values[name] for name in self._graph[node_to_compute]]
            if len(input) == 1:
                input = input[0]  # unpack iterable if possible
            module = self._modules[node_to_compute]
            output = module(input)

            # we may get one output:
            output_names = getattr(module, "_output_names", None)
            if output_names is None or len(output_names) == 1:
                if output_names is not None:
                    assert output_names[0] == node_to_compute, "invalid graph"
                values[node_to_compute] = output
                _mark_as_computed(node_to_compute)

            # or multiple outputs:
            else:
                assert isinstance(
                    output, tuple
                ), f"expected outputs {output_names} of {module} to be tuple, not {type(output)}"
                assert (
                    len(output_names) == len(output)
                ), f"expected {len(output_names)} outputs from {module}, received {len(output)}"
                for node, value in zip(output_names, output):
                    values[node] = value
                    _mark_as_computed(node)

            # return output if it is available:
            if all(computed[output_name] for output_name in self.output_names):
                result = [values[output_name] for output_name in self.output_names]
                return result[0] if len(result) == 1 else tuple(result)

            # find next node to compute:
            node_to_compute = _find_computable_node()

            # clean up values we no longer need:
            _clear_unused_values()

        # this should never happen:
        raise ValueError("nn.Graph.forward() failed. Is graph unconnected?")

    def to_pytorch(self):
        if not hasattr(self, "pytorch_model"):
            raise AttributeError("Shardwise Graph detached from PyTorch model.")
        if self.encrypted:
            raise ValueError(
                "Shardwise model must be decrypted before calling to_pytorch()"
            )
        with torch.no_grad():
            for name, param in self.pytorch_model.named_parameters():
                param.set_(self._modules[name].data)
        return self.pytorch_model


class Sequential(Graph):
    """
    Sequence of modules.
    """

    def __init__(self, *module_list, input_names=None):
        if input_names is None:
            input_names = ["input"]
        super().__init__(input_names, "output")
        if len(module_list) == 1 and isinstance(module_list[0], list):
            raise DeprecationWarning(
                "passing shardwise.nn.Sequential a list is deprecated. Please "
                "pass unpacked arguments (e.g. Sequential(*my_modules))."
            )
            module_list = module_list[0]

        for idx, module in enumerate(module_list):
            if isinstance(module, OrderedDict):
                for key, val in module.items():
                    self.add_module(key, val, input_names)
                    input_names = key
                self.output_names = [key]
            else:
                module_name = str(idx)
                if idx > 0:
                    input_names = [str(idx - 1)]
                self.add_module(module_name, module, input_names)
                self.output_names = [module_name]


class ModuleList(Module):
    r"""Holds submodules in a list.

    :class:`~shardwise.nn.ModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~shardwise.nn.Module` methods.

    Args:
        modules (iterable, optional): an iterable of modules to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    def __init__(self, modules=None):
        super(ModuleList, self).__init__()
        if modules is not None:
            self += modules

    def __dir__(self):
        keys = super(ModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                del self._modules[str(k)]
        else:
            del self._modules[self._get_abs_string_index(idx)]
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))


__module_list_func_names = [
    "_get_abs_string_index",
    "__getitem__",
    "__setitem__",
    "__len__",
    "__iter__",
    "__iadd__",
    "insert",
    "append",
    "extend",
]


for func_name in __module_list_func_names:
    func = getattr(torch.nn.ModuleList, func_name)
    setattr(ModuleList, func_name, func)


class ModuleDict(Module):
    r"""Holds submodules in a dictionary.

    :class:`~shardwise.nn.ModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~shardwise.nn.Module` methods.

    :class:`~shardwise.nn.ModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~shardwise.nn.ModuleDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~shardwise.nn.ModuleDict` (the argument to :meth:`~shardwise.nn.ModuleDict.update`).

    Note that :meth:`~shardwise.nn.ModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Arguments:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.choices = nn.ModuleDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    def __init__(self, modules=None):
        super(ModuleDict, self).__init__()
        if modules is not None:
            self.update(modules)


__module_dict_func_names = [
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "__len__",
    "__iter__",
    "__contains__",
    "clear",
    "pop",
    "keys",
    "items",
    "values",
    "update",
    "forward",
]


for func_name in __module_dict_func_names:
    func = getattr(torch.nn.ModuleDict, func_name)
    setattr(ModuleDict, func_name, func)


class Parameter(Module):
    """
    Module that holds a parameter tensor. If `trainable` is set to `False`, the
    parameter will be treated as a buffer: it will not be trained. Hence,
    parameters do not inherit `requires_grad` from the tensor they are initialized
    with.

    Parameters are encrypted when the `encrypt()` function is called on a module.
    """

    def __init__(self, param, trainable=True):
        super().__init__()

        # ensure that input is a PyTorch or Shardwise tensor:
        assert torch.is_tensor(param) or shardwise.is_encrypted_tensor(
            param
        ), f"param must be PyTorch of Shardwise tensor, not {type(param)}"
        if isinstance(param, torch.nn.parameter.Parameter):
            param = param.data

        # register parameter or buffer:
        if trainable:
            self.register_parameter("data", param)
        else:
            self.register_buffer("data", param)

        # register whether or not module is encrypted:
        self.encrypted = shardwise.is_encrypted_tensor(param)

    def forward(self, input):
        return self.data

    @property
    def requires_grad(self):
        return self.data.requires_grad


class Constant(Module):
    """
    Module that holds a constant tensor. If `trainable` is set to `False`, the
    parameter will be treated as a buffer: it will not be trained.

    Constants are not encrypted when the `encrypt()` function is called on a module.
    """

    SUPPORTS_PLAINTEXT_INPUTS = True

    def __init__(self, value):
        super().__init__()
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        assert torch.is_tensor(
            value
        ), f"value must be PyTorch tensor, not {type(value)}"
        self.value = value.to(dtype=torch.float)

    def forward(self, input):
        return self.value

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        assert "value" in attributes, "No value for Constant specified."
        return Constant(attributes["value"])

    def encrypt(self, mode=True, src=0):
        self.encrypted = mode
        return self


class ConstantOfShape(Module):
    """
    Modules that returns a matrix of specified size containing a constant.
    """

    SUPPORTS_PLAINTEXT_INPUTS = True

    def __init__(self, value):
        super().__init__()
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        assert torch.is_tensor(
            value
        ), f"value must be PyTorch tensor, not {type(value)}"
        self.value = value.to(dtype=torch.float)

    def forward(self, size):
        if torch.is_tensor(size):
            size = size.int().tolist()
        assert isinstance(
            size, (list, tuple)
        ), f"size must be list or tuple, not {type(size)}"
        return self.value.expand(*size)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        assert "value" in attributes, "No value for ConstantOfShape specified."
        return ConstantOfShape(attributes["value"])

    def encrypt(self, mode=True, src=0):
        self.encrypted = mode
        return self
