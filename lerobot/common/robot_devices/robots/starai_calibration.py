# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logic to calibrate a robot arm built with dynamixel motors"""
# TODO(rcadene, aliberts): move this logic into the robot code when refactoring

import numpy as np

from lerobot.common.robot_devices.motors.starai import (
    CalibrationMode,
    TorqueMode,
)
from lerobot.common.robot_devices.motors.utils import MotorsBus

URL_TEMPLATE = (
    "https://raw.githubusercontent.com/huggingface/lerobot/main/media/{robot}/{arm}_{position}.webp"
)

# The following positions are provided in nominal degree range ]-180, +180[
# For more info on these constants, see comments in the code where they get used.
ZERO_POSITION_DEGREE = 0
ROTATED_POSITION_DEGREE = 0.1


def assert_drive_mode(drive_mode):
    # `drive_mode` is in [0,1] with 0 means original rotation direction for the motor, and 1 means inverted.
    if not np.all(np.isin(drive_mode, [0, 1])):
        raise ValueError(f"`drive_mode` contains values other than 0 or 1: ({drive_mode})")


def apply_drive_mode(position, drive_mode):
    assert_drive_mode(drive_mode)
    # Convert `drive_mode` from [0, 1] with 0 indicates original rotation direction and 1 inverted,
    # to [-1, 1] with 1 indicates original rotation direction and -1 inverted.
    signed_drive_mode = -(drive_mode * 2 - 1)
    position *= signed_drive_mode
    return position


def compute_nearest_rounded_position(position, models):
    delta_turn = ROTATED_POSITION_DEGREE
    nearest_pos = np.round(position.astype(float) / delta_turn) * delta_turn
    return nearest_pos.astype(position.dtype)


def run_arm_calibration(arm: MotorsBus, robot_type: str, arm_name: str, arm_type: str):

    # if (arm.read("Torque_Enable") != TorqueMode.DISABLED.value).any():
    #     raise ValueError("To run calibration, the torque must be disabled on all motors.")

    print(f"\nRunning calibration of {robot_type} {arm_name} {arm_type}...")

    print("\nMove arm to zero position")
    # print("See: " + URL_TEMPLATE.format(robot=robot_type, arm=arm_type, position="zero"))
    input("Press Enter to continue...")
    print()

 
    # Compute homing offset so that `present_position + homing_offset ~= target_position`.
    zero_pos = arm.read("Present_Position")
    # zero_nearest_pos = compute_nearest_rounded_position(zero_pos, arm.motor_models)
    # for i in zero_pos:
    #     zero_pos[i] = -zero_pos[i]
    homing_offset = -zero_pos
    print(homing_offset)


    print("\nMove arm to open position")
    # print("See: " + URL_TEMPLATE.format(robot=robot_type, arm=arm_type, position="open"))
    input("Press Enter to continue...")
    print()
    start_pos = arm.read("Present_Position")
    print(start_pos)


    print("\nMove arm to rest position")
    # print("See: " + URL_TEMPLATE.format(robot=robot_type, arm=arm_type, position="rest"))
    input("Press Enter to continue...")
    print()
    rest_pos = arm.read("Present_Position")
    print(rest_pos)


    calib_mode = []
    for name in arm.motor_names:
        if name == "gripper":
            calib_mode.append(CalibrationMode.LINEAR.name)
        else:
            calib_mode.append(CalibrationMode.DEGREE.name)

    calib_data = {
        "homing_offset": homing_offset.tolist(),
        "start_pos": start_pos.tolist(),
        "end_pos": rest_pos.tolist(),
        "calib_mode": calib_mode,
        "motor_names": arm.motor_names,
    }
    return calib_data
