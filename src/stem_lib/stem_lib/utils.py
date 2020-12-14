from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
import ros2param.api
import time
from .stdlib.stopwatch import Stopwatch


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


class StateChangeListener:
    def __init__(self):
        self._last_update_time = None
        self._current_state = None
        self._last_state = None
        self._has_changed = False

    def has_changed(self, timeout=float('inf')):
        if self._last_update_time is None:
            return False, None

        if (time.time() - self._last_update_time > timeout
            or self._has_changed):
            self._has_changed = False
            return True, self._last_state
        
        return False, None


    def update(self, state):
        
        # except first state update
        if (self._last_update_time is not None
            and self._current_state != state):

            self._last_state = self._current_state
            self._has_changed = True

        self._current_state = state
        self._last_update_time = time.time()


class SamplingRateWatchdog:
    def __init__(self, sampling_rate_min):
        self._sampling_rate_min = sampling_rate_min
        self._sw = Stopwatch()
        self._sampling_rate_average = 0

    @property
    def sampling_rate(self):
        return self._sampling_rate_average

    def start(self):
        self._sw.start()
    
    def lap(self):
        rate = 1 / (self._sw.lap() + 1e-8)
        self._sampling_rate_average = 0.5 * (rate + self._sampling_rate_average)
        return self._sampling_rate_average >= self._sampling_rate_min
