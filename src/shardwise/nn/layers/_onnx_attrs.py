"""Helpers for reading ONNX node attributes."""


def _all_the_same(items):
    """
    Checks whether all values in a list are the same.
    """
    return all(items[0] == item for item in items)


def _identify_bool_attributes_with_defaults(attributes, attr_name, attr_value, default=True):
    """For boolean attributes that have default values in the ONNX specification
    checks to see if they are present in `attributes`, and assigns the
    default if not present and appropriate value if present. Note `attr_value`
    must be the value `attributes[attr_name]` if the default is to be kept.
    """
    output = default
    if attr_name in attributes and attributes[attr_name] != attr_value:
        output = not default
    return output
