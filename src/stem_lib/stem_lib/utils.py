from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
import ros2param.api

def load_parameter(node, parameter_name, default_value):
    node.declare_parameter(parameter_name, default_value)
    return node.get_parameter(parameter_name).value

def fill_message_from_dict(message, fields):
    for name, value in fields.items():
        exec('message.' + name + '=value')
    return message

def request_get_parameters_async(client, parameter_names):
    request = GetParameters.Request()
    request.names = parameter_names

    return request_service_async(client, request)

def request_service_async(client, request, timeout_sec=0.25):
    if not client.service_is_ready():
        ready = client.wait_for_service(timeout_sec)
        if not ready:
            raise RuntimeError('Wait for service timed out')
    
    return client.call_async(request)

def get_parameter_value(parameter_value, allowed_types):
    """
    Parameters
    ----------
        parameter_value : rcl_interfaces.msg.ParameterValue
        allowed_types : list

    """
    if parameter_value.type not in allowed_types:
        raise RuntimeError(f'ValidationError. parameter type {parameter_value.type} is not in allowed_types {allowed_types}')

    return ros2param.api.get_value(parameter_value=parameter_value)