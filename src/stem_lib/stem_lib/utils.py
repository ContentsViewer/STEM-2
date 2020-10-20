
def load_parameter(node, parameter_name, default_value):
    node.declare_parameter(parameter_name, default_value)
    return node.get_parameter(parameter_name).value

def fill_message_from_dict(message, fields):
    for name, value in fields.items():
        exec('message.' + name + '=value')
    return message