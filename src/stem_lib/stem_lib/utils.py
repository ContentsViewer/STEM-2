
def load_parameter(node, parameter_name, default_value):
    node.declare_parameter(parameter_name, default_value)
    return node.get_parameter(parameter_name).value
    