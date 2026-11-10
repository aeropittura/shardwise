#!/usr/bin/env python3


import copy
import io

import onnx
from onnx import numpy_helper
import torch
from torch.onnx import OperatorExportTypes
import torch.onnx.symbolic_helper as sym_help
import torch.onnx.utils

from shardwise.nn import layers

try:
    import tensorflow as tf
    import tf2onnx

    TF_AND_TF2ONNX = True
except ImportError:
    TF_AND_TF2ONNX = False

try:
    import torch.onnx.symbolic_registry as sym_registry  # noqa

    SYM_REGISTRY = True
except ImportError:
    from torch.onnx._internal.registration import registry  # noqa

    SYM_REGISTRY = False


_OPSET_VERSION = 17


def from_onnx(onnx_string_or_file):
    """
    Converts an ONNX model serialized in an `onnx_string_or_file` to a Shardwise model.
    """
    onnx_model = _load_onnx_model(onnx_string_or_file)
    return _to_shardwise(onnx_model)


def from_pytorch(pytorch_model, dummy_input):
    """
    Converts a PyTorch model `pytorch_model` into a Shardwise model by tracing it
    using the input `dummy_input`.
    """

    # construct Shardwise model:
    f = _from_pytorch_to_bytes(pytorch_model, dummy_input)
    shardwise_model = from_onnx(f)
    f.close()

    # set model architecture to export model back to pytorch model
    shardwise_model.pytorch_model = copy.deepcopy(pytorch_model)

    # make sure training / eval setting is copied:
    shardwise_model.train(mode=pytorch_model.training)
    return shardwise_model


def from_tensorflow(tensorflow_graph_def, inputs, outputs):
    """
    Function that converts Tensorflow model into Shardwise model based on
    https://github.com/onnx/tensorflow-onnx/blob/master/tf2onnx/convert.py
    The model is returned in evaluation mode.
    Args:
        `tensorflow_graph_def`: Input Tensorflow GraphDef to be converted
        `inputs`: input nodes
        `outputs`: output nodes
    """
    raise DeprecationWarning(
        "shardwise.nn.from_tensorflow is deprecated. ",
        "Shardwise will no longer support model conversion from TensorFlow.",
    )
    # Exporting model to ONNX graph
    if not TF_AND_TF2ONNX:
        raise ImportError("Please install both tensorflow and tf2onnx packages")

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(tensorflow_graph_def, name="")
    with tf2onnx.tf_loader.tf_session(graph=tf_graph):
        g = tf2onnx.tfonnx.process_tf_graph(
            tf_graph,
            opset=10,
            continue_on_error=False,
            input_names=inputs,
            output_names=outputs,
        )
    onnx_graph = tf2onnx.optimizer.optimize_graph(g)
    model_proto = onnx_graph.make_model(f"converted from {tensorflow_graph_def}")
    f = io.BytesIO()
    f.write(model_proto.SerializeToString())

    # construct Shardwise model
    # Note: We don't convert shardwise model to training mode, as Tensorflow
    # models are used for both training and evaluation without the specific
    # conversion of one mode to another
    f.seek(0)
    shardwise_model = from_onnx(f)
    return shardwise_model


def _from_pytorch_to_bytes(pytorch_model, dummy_input):
    """
    Returns I/O stream containing ONNX graph for `pytorch_model` traced with
    input `dummy_input`.
    """

    # first export is only used to obtain the PyTorch-to-ONNX symbolic registry:
    with io.BytesIO() as f:
        _export_pytorch_model(f, pytorch_model, dummy_input)

    # update ONNX symbolic registry with Shardwise-specific functions:
    _update_onnx_symbolic_registry()

    # export again so the graph is created with Shardwise-specific registry:
    f = io.BytesIO()
    f = _export_pytorch_model(f, pytorch_model, dummy_input)
    f.seek(0)
    return f


def _export_pytorch_model(f, pytorch_model, dummy_input):
    """
    Returns a binary I/O stream containing ONNX-exported pytorch_model that was
    traced with input `dummy_input`.
    """
    kwargs = {
        "do_constant_folding": False,
        "export_params": True,
        "input_names": ["input"],
        "operator_export_type": OperatorExportTypes.ONNX,
        "output_names": ["output"],
        "opset_version": _OPSET_VERSION,
    }
    torch.onnx.export(pytorch_model, dummy_input, f, **kwargs)
    return f


# mapping from ONNX to shardwise.nn for modules with different names:
ONNX_TO_SHARDWISE = {
    "adaptive_avg_pool2d": layers.AdaptiveAvgPool2d,
    "adaptive_max_pool2d": layers.AdaptiveMaxPool2d,
    "AveragePool": layers.AvgPool2d,
    "Clip": layers.Hardtanh,
    "MaxPool": layers.MaxPool2d,
    "Pad": layers._ConstantPad,
    "Relu": layers.ReLU,
    "ReduceMean": layers.Mean,
    "ReduceSum": layers.Sum,
}


def _to_shardwise(onnx_model):
    """
    Function that converts an `onnx_model` to a Shardwise model.
    """

    # create graph:
    input_names, output_names = _get_input_output_names(onnx_model)
    assert len(output_names) == 1, "Only one output per model supported."
    shardwise_model = layers.Graph(input_names, output_names[0])

    # create nodes for the parameters:
    for node in onnx_model.graph.initializer:
        param = torch.from_numpy(numpy_helper.to_array(node))
        shardwise_model.add_module(node.name, layers.Parameter(param), [])

    # loop over all nodes:
    for node in onnx_model.graph.node:
        # get attributes and node type:
        attributes = {attr.name: _get_attribute_value(attr) for attr in node.attribute}
        shardwise_class = _get_operator_class(node.op_type, attributes)

        # add Shardwise module to graph:
        shardwise_module = shardwise_class.from_onnx(attributes=attributes)
        input_names = list(node.input)
        output_names = list(node.output)
        if node.op_type == "Dropout":
            output_names = [output_names[0]]  # do not output Dropout mask
        shardwise_model.add_module(
            output_names[0], shardwise_module, input_names, output_names=output_names
        )

    # return final model:
    shardwise_model = _get_model_or_module(shardwise_model)
    return shardwise_model


def _load_onnx_model(onnx_string_or_file):
    """
    Loads ONNX model from file or string.
    """
    if hasattr(onnx_string_or_file, "seek"):
        onnx_string_or_file.seek(0)
        return onnx.load(onnx_string_or_file)
    return onnx.load_model_from_string(onnx_string_or_file)


def _get_input_output_names(onnx_model):
    """
    Return input and output names of the ONNX graph.
    """
    input_names = [input.name for input in onnx_model.graph.input]
    output_names = [output.name for output in onnx_model.graph.output]
    assert len(input_names) >= 1, "number of inputs should be at least 1"
    assert len(output_names) == 1, "number of outputs should be 1"
    return input_names, output_names


def _get_model_or_module(shardwise_model):
    """
    Returns `Module` if model contains only one module. Otherwise returns model.
    """
    num_modules = len(list(shardwise_model.modules()))
    if num_modules == 1:
        for shardwise_module in shardwise_model.modules():
            return shardwise_module
    return shardwise_model


def _get_attribute_value(attr):
    """
    Retrieves value from an ONNX attribute.
    """
    if attr.HasField("f"):  # floating-point attribute
        return attr.f
    elif attr.HasField("i"):  # integer attribute
        return attr.i
    elif attr.HasField("s"):  # string attribute
        return attr.s  # TODO: Sanitize string.
    elif attr.HasField("t"):  # tensor attribute
        return torch.from_numpy(numpy_helper.to_array(attr.t))
    elif len(attr.ints) > 0:
        return list(attr.ints)
    elif len(attr.floats) > 0:
        return list(attr.floats)
    raise ValueError(f"Unknown attribute type for attribute {attr.name}.")


def _get_operator_class(node_op_type, attributes):
    """
    Returns the `shardwise.nn.Module` type corresponding to an ONNX node.
    """
    shardwise_class = getattr(layers, node_op_type, ONNX_TO_SHARDWISE.get(node_op_type, None))
    if shardwise_class is None:
        raise ValueError(f"Shardwise does not support ONNX op {node_op_type}.")
    return shardwise_class


def _update_onnx_symbolic_registry():
    """
    Updates the ONNX symbolic registry for operators that need a Shardwise-specific
    implementation and custom operators.
    """
    if SYM_REGISTRY:
        # update PyTorch's symbolic ONNX registry to output different functions:
        for version_key, version_val in sym_registry._registry.items():
            for function_key in version_val.keys():
                if function_key == "softmax":
                    sym_registry._registry[version_key][function_key] = _onnx_shardwise_softmax
                if function_key == "log_softmax":
                    sym_registry._registry[version_key][function_key] = _onnx_shardwise_logsoftmax
                if function_key == "dropout":
                    sym_registry._registry[version_key][function_key] = _onnx_shardwise_dropout
                if function_key == "feature_dropout":
                    sym_registry._registry[version_key][function_key] = (
                        _onnx_shardwise_feature_dropout
                    )
    else:
        # Update ONNX symbolic registry using torch.onnx.register_custom_op_symbolic
        torch.onnx.register_custom_op_symbolic(
            "aten::softmax", _onnx_shardwise_softmax, _OPSET_VERSION
        )
        torch.onnx.register_custom_op_symbolic(
            "aten::log_softmax", _onnx_shardwise_logsoftmax, _OPSET_VERSION
        )
        torch.onnx.register_custom_op_symbolic(
            "aten::dropout", _onnx_shardwise_dropout, _OPSET_VERSION
        )
        torch.onnx.register_custom_op_symbolic(
            "aten::feature_dropout", _onnx_shardwise_feature_dropout, _OPSET_VERSION
        )


@sym_help.parse_args("v", "i", "none")
def _onnx_shardwise_softmax(g, input, dim, dtype=None):
    """
    This function converts PyTorch's Softmax module to a Softmax module in
    the ONNX model. It overrides PyTorch's default conversion of Softmax module
    to a sequence of Exp, ReduceSum and Div modules, since this default
    conversion can cause numerical overflow when applied to ShardTensors.
    """
    result = g.op("Softmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = sym_help._get_const(dtype, "i", "dtype")
        result = g.op("Cast", result, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    return result


@sym_help.parse_args("v", "i", "none")
def _onnx_shardwise_logsoftmax(g, input, dim, dtype=None):
    """
    This function converts PyTorch's LogSoftmax module to a LogSoftmax module in
    the ONNX model. It overrides PyTorch's default conversion of LogSoftmax module
    to avoid potentially creating Transpose operators.
    """
    result = g.op("LogSoftmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = sym_help._get_const(dtype, "i", "dtype")
        result = g.op("Cast", result, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    return result


@sym_help.parse_args("v", "f", "i")
def _onnx_shardwise_dropout(g, input, p, train):
    """
    This function converts PyTorch's Dropout module to a Dropout module in the ONNX
    model. It overrides PyTorch's default implementation to ignore the Dropout module
    during the conversion. PyTorch assumes that ONNX models are only used for
    inference and therefore Dropout modules are not required in the ONNX model.
    However, Shardwise needs to convert ONNX models to trainable
    Shardwise models, and so the Dropout module needs to be included in the
    Shardwise-specific conversion.
    """
    r, _ = g.op("Dropout", input, ratio_f=p, outputs=2)
    return r


@sym_help.parse_args("v", "f", "i")
def _onnx_shardwise_feature_dropout(g, input, p, train):
    """
    This function converts PyTorch's DropoutNd module to a DropoutNd module in the ONNX
    model. It overrides PyTorch's default implementation to ignore the DropoutNd module
    during the conversion. PyTorch assumes that ONNX models are only used for
    inference and therefore DropoutNd modules are not required in the ONNX model.
    However, Shardwise needs to convert ONNX models to trainable
    Shardwise models, and so the DropoutNd module needs to be included in the
    Shardwise-specific conversion.
    """
    r, _ = g.op("DropoutNd", input, ratio_f=p, outputs=2)
    return r
