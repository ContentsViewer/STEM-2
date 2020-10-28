from rcl_interfaces.srv import GetParameters

def load_parameter(node, parameter_name, default_value):
    node.declare_parameter(parameter_name, default_value)
    return node.get_parameter(parameter_name).value

def fill_message_from_dict(message, fields):
    for name, value in fields.items():
        exec('message.' + name + '=value')
    return message

def call_get_parameters_async(*, node, node_name, parameter_names):
    # create client
    client = node.create_client(GetParameters, f'{node_name}/get_parameters')

    # call as soon as ready
    ready = client.wait_for_service(timeout_sec=5.0)
    if not ready:
        raise RuntimeError('Wait for service timed out')

    request = GetParameters.Request()
    request.names = parameter_names
    return client.call_async(request)
    