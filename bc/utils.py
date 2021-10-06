import numpy as np
import torch

MAX_THROTTLE = 20
MAX_BRAKE = 20
TARGET_SPEED_RANGE = (0, 20)
STEER_RANGE = (-0.6, 0.6)


def range_normalization(value: float, maximum: float, minimum: float, factor: float = 1):
    """Variable linear normalization to range [-factor, factor] in numpy

    Args:
        value (float): Unnormalized variable
        maximum (float): Maximum value
        minimum (float): Minimum value
        factor (float, optional): Range Scale. Defaults to 1.

    Returns:
        float: Normalized variable
    """
    value_traslation = (minimum + maximum) / 2
    value_factor = np.abs(maximum - value_traslation)
    value = np.clip(value, minimum, maximum)
    value = (value - value_traslation) / value_factor
    return value * factor


def torch_range_normalization(value: torch.Tensor, maximum: float, minimum: float, factor: float = 1) -> torch.Tensor:
    """Variable linear normalization to range [-factor, factor] in torch

    Args:
        value (float): Unnormalized variable
        maximum (float): Maximum value
        minimum (float): Minimum value
        factor (float, optional): Range Scale. Defaults to 1.

    Returns:
        float: Normalized variable
    """
    value_translation = (minimum + maximum) / 2
    value_factor = abs(maximum - value_translation)
    value = torch.clip(value, minimum, maximum)
    value = (value - value_translation) / value_factor
    return value * factor


def range_unnormalization(value: float, maximum: float, minimum: float, factor: float = 1):
    """Variable linear unnormalization from range [-factor, factor] in numpy

    Args:
        value (float): Normalized variable
        maximum (float): Maximum value
        minimum (float): Minimum value
        factor (float, optional): Range Scale. Defaults to 1.

    Returns:
        float: Unnormalized variable
    """
    value_translation = (minimum + maximum) / 2
    value_factor = np.abs(maximum - value_translation)
    value = np.clip(value, -1, 1)
    value = value_factor * value / factor + value_translation
    return value


def torch_range_unnormalization(value: torch.Tensor, maximum: float, minimum: float, factor: float = 1) -> torch.Tensor:
    """Variable linear unnormalization from range [-factor, factor] in torch

    Args:
        value (float): Normalized variable
        maximum (float): Maximum value
        minimum (float): Minimum value
        factor (float, optional): Range Scale. Defaults to 1.

    Returns:
        float: Unnormalized variable
    """
    value_translation = (minimum + maximum) / 2
    value_factor = abs(maximum - value_translation)
    value = torch.clip(value, -1, 1)
    value = value_factor * value / factor + value_translation
    return value


def normalize_action(action: list):
    throttle, brake, steer = float(action[0]), float(action[1]), float(action[2])

    # Throttle and Brake Normalization, always positive
    throttle = np.abs(throttle) / MAX_THROTTLE
    brake = np.abs(brake) / MAX_BRAKE

    return throttle, brake, steer


def unnormalize_action(action: list):
    throttle, brake, steer = float(action[0]), float(action[1]), float(action[2])
    # Throttle and Brake UnNormalization
    throttle = np.abs(throttle) * MAX_THROTTLE
    brake = np.abs(brake) * MAX_BRAKE

    return throttle, brake, steer



def normalize_pid_action(action: list):
    target_speed, steer = float(action[0]), float(action[1])
    # Target Speed normalization from [0, 20] to [-1, 1]
    target_speed = range_normalization(target_speed, TARGET_SPEED_RANGE[1], TARGET_SPEED_RANGE[0], 1)
    return steer, target_speed


def unnormalize_pid_action(action: list):
    target_speed, steer = float(action[0]), float(action[1])
    target_speed = range_unnormalization(target_speed, TARGET_SPEED_RANGE[1], TARGET_SPEED_RANGE[0], 1)
    return steer, target_speed


def unnormalize_pid_action_torch(action: torch.Tensor) -> torch.Tensor:
    """
    Unnormalize the action. The target speed goes from [-1, 1] to [-0.6, 0.6] linearly.
    """
    target_speed, steer = action[:, 0], action[:, 1]
    target_speed = torch_range_unnormalization(target_speed, TARGET_SPEED_RANGE[1], TARGET_SPEED_RANGE[0], 1)
    steer = torch_range_unnormalization(steer, STEER_RANGE[1], STEER_RANGE[0], 1)
    return torch.stack([target_speed, steer], dim=1)
