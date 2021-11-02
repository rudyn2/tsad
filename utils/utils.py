import numpy as np
import torch

INPUT_THROTTLE_RANGE = (0, 1)
THROTTLE_RANGE = (0, 1)

INPUT_BRAKE_RANGE = (0, 1)
BRAKE_RANGE = (0, 1)

INPUT_STEER_RANGE = (-0.6, 0.6)
STEER_RANGE = (-0.5, 0.5)

INPUT_TARGET_SPEED_RANGE = (0, 20)
TARGET_SPEED_RANGE = (0, 1)

def range_transformation(value: float, input_range: list, output_range: list = [-1, 1]):
    """Range transformation from [input_range[0], input_range[1]] to [output_range[0], output_range[1]]

    Args:
        value (float): variable
        input_range (list): desired input variable range
        output_range (list, optional): desired output range. Defaults to [-1, 1].

    Returns:
        float: variable transformed
    """    
    assert input_range[0] < input_range[1]
    assert output_range[0] < output_range[1]

    input_offset = (input_range[1] + input_range[0]) / 2
    input_factor = np.abs(input_range[1] - input_range[0]) / 2

    value = np.clip(value, input_range[0], input_range[1])
    # Scaling to [-1, 1]
    value = (value - input_offset) / input_factor

    output_offset = (output_range[1] + output_range[0]) / 2
    output_factor = np.abs(output_range[1] - output_range[0]) / 2
    
    return value * output_factor + output_offset

def normalize_action(action: list):
    if len(action) == 2:
        target_speed, steer = float(action[0]), float(action[1])
        target_speed = range_transformation(target_speed, input_range=INPUT_TARGET_SPEED_RANGE, output_range=TARGET_SPEED_RANGE)
        steer = range_transformation(steer, input_range=INPUT_STEER_RANGE, output_range=STEER_RANGE)
        return target_speed, steer
    elif len(action) == 3:
        throttle, brake, steer = float(action[0]), float(action[1]), float(action[2])
        throttle = range_transformation(throttle, input_range=INPUT_THROTTLE_RANGE, output_range=THROTTLE_RANGE)
        brake = range_transformation(brake, input_range=INPUT_BRAKE_RANGE, output_range=BRAKE_RANGE)
        steer = range_transformation(steer, input_range=INPUT_STEER_RANGE, output_range=STEER_RANGE)
        return throttle, brake, steer
    else:
        raise Exception('Wrong action dimension')

def normalize_speed(speed: float):
    return range_transformation(speed, input_range=INPUT_TARGET_SPEED_RANGE, output_range=TARGET_SPEED_RANGE)

def unnormalize_action(action: list):
    if len(action) == 2:
        target_speed, steer = float(action[0]), float(action[1])
        target_speed = range_transformation(target_speed, input_range=TARGET_SPEED_RANGE, output_range=INPUT_TARGET_SPEED_RANGE)
        steer = range_transformation(steer, input_range=STEER_RANGE, output_range=INPUT_STEER_RANGE)
        return target_speed, steer
    elif len(action) == 3:
        throttle, brake, steer = float(action[0]), float(action[1]), float(action[2])
        throttle = range_transformation(throttle, input_range=THROTTLE_RANGE, output_range=INPUT_THROTTLE_RANGE)
        brake = range_transformation(brake, input_range=BRAKE_RANGE, output_range=INPUT_BRAKE_RANGE)
        steer = range_transformation(steer, input_range=STEER_RANGE, output_range=INPUT_STEER_RANGE)
        return throttle, brake, steer
    else:
        raise Exception('Wrong action dimension')

def get_collate_fn(act_mode: str):
    def collate_fn(samples: list) -> (dict, dict):
        """
        Returns a dictionary with grouped samples. Each sample is a tuple which comes from the dataset __getitem__.
        """
        obs, act = [], []
        for t in samples:
            obs.append(dict(encoding=t[0]))
            if act_mode == "pid":
                act.append(np.array([t[2], t[1][2]]))
            else:
                act.append(t[1])
        return obs, act
    return collate_fn