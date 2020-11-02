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

    request = GetParameters.Request()
    request.names = parameter_names
    return request_service_async(client, request, 5.0)

def request_service_async(client, request, timeout_sec=None):
    if not client.service_is_ready():
        ready = client.wait_for_service(timeout_sec)
        if not ready:
            raise RuntimeError('Wait for service timed out')
    
    return client.call_async(request)