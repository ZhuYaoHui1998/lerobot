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

import enum
import logging
import math
import time
import traceback
from copy import deepcopy
import struct

import numpy as np
import tqdm

from lerobot.common.robot_devices.motors.configs import StaraiMotorsBusConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc
import fashionstar_uart_sdk as uservo
import serial


BAUDRATE = 1_000_000
TIMEOUT_MS = 1000

MAX_ID_RANGE = 252

# The following bounds define the lower and upper joints range (after calibration).
# For joints in degree (i.e. revolute joints), their nominal range is [-180, 180] degrees
# which corresponds to a half rotation on the left and half rotation on the right.
# Some joints might require higher range, so we allow up to [-270, 270] degrees until
# an error is raised.
LOWER_BOUND_DEGREE = -360
UPPER_BOUND_DEGREE = 360
# For joints in percentage (i.e. joints that move linearly like the prismatic joint of a gripper),
# their nominal range is [0, 100] %. For instance, for Aloha gripper, 0% is fully
# closed, and 100% is fully open. To account for slight calibration issue, we allow up to
# [-10, 110] until an error is raised.
LOWER_BOUND_LINEAR = -140
UPPER_BOUND_LINEAR = 140
HALF_TURN_DEGREE = 90
# https://emanual.robotis.com/docs/en/dxl/x/xl330-m077
# https://emanual.robotis.com/docs/en/dxl/x/xl330-m288
# https://emanual.robotis.com/docs/en/dxl/x/xl430-w250
# https://emanual.robotis.com/docs/en/dxl/x/xm430-w350
# https://emanual.robotis.com/docs/en/dxl/x/xm540-w270
# https://emanual.robotis.com/docs/en/dxl/x/xc430-w150

# data_name: (address, size_byte)

U_SERIES_BAUDRATE_TABLE = {
    0: 115_200,
    1: 1_000_000,
}

CALIBRATION_REQUIRED = ["Goal_Position", "Present_Position"]
CONVERT_UINT32_TO_INT32_REQUIRED = [""]
COMM_SUCCESS = 128



NUM_READ_RETRY = 10
NUM_WRITE_RETRY = 10


# def convert_degrees_to_steps(degrees: float | np.ndarray, models: str | list[str]) -> np.ndarray:
#     """This function converts the degree range to the step range for indicating motors rotation.
#     It assumes a motor achieves a full rotation by going from -180 degree position to +180.
#     The motor resolution (e.g. 4096) corresponds to the number of steps needed to achieve a full rotation.
#     """
#     resolutions = [MODEL_RESOLUTION[model] for model in models]
#     steps = degrees / 180 * np.array(resolutions) / 2
#     steps = steps.astype(int)
#     return steps


# def convert_to_bytes(value, bytes, mock=False):
#     if mock:
#         return value

#     import dynamixel_sdk as dxl

#     # Note: No need to convert back into unsigned int, since this byte preprocessing
#     # already handles it for us.
#     if bytes == 1:
#         data = [
#             dxl.DXL_LOBYTE(dxl.DXL_LOWORD(value)),
#         ]
#     elif bytes == 2:
#         data = [
#             dxl.DXL_LOBYTE(dxl.DXL_LOWORD(value)),
#             dxl.DXL_HIBYTE(dxl.DXL_LOWORD(value)),
#         ]F
#     elif bytes == 4:
#         data = [
#             dxl.DXL_LOBYTE(dxl.DXL_LOWORD(value)),
#             dxl.DXL_HIBYTE(dxl.DXL_LOWORD(value)),
#             dxl.DXL_LOBYTE(dxl.DXL_HIWORD(value)),
#             dxl.DXL_HIBYTE(dxl.DXL_HIWORD(value)),
#         ]
#     else:
#         raise NotImplementedError(
#             f"Value of the number of bytes to be sent is expected to be in [1, 2, 4], but "
#             f"{bytes} is provided instead."
#         )
#     return data


def get_group_sync_key(data_name, motor_names):
    group_key = f"{data_name}_" + "_".join(motor_names)
    return group_key


def get_result_name(fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    rslt_name = f"{fn_name}_{group_key}"
    return rslt_name


def get_queue_name(fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    queue_name = f"{fn_name}_{group_key}"
    return queue_name


def get_log_name(var_name, fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    log_name = f"{var_name}_{fn_name}_{group_key}"
    return log_name


# def assert_same_address(model_ctrl_table, motor_models, data_name):
#     all_addr = []
#     all_bytes = []
#     for model in motor_models:
#         addr, bytes = model_ctrl_table[model][data_name]
#         all_addr.append(addr)
#         all_bytes.append(bytes)

#     if len(set(all_addr)) != 1:
#         raise NotImplementedError(
#             f"At least two motor models use a different address for `data_name`='{data_name}' ({list(zip(motor_models, all_addr, strict=False))}). Contact a LeRobot maintainer."
#         )

#     if len(set(all_bytes)) != 1:
#         raise NotImplementedError(
#             f"At least two motor models use a different bytes representation for `data_name`='{data_name}' ({list(zip(motor_models, all_bytes, strict=False))}). Contact a LeRobot maintainer."
#         )


class TorqueMode(enum.Enum):
    ENABLED = 1
    DISABLED = 0


class DriveMode(enum.Enum):
    NON_INVERTED = 0
    INVERTED = 1


class CalibrationMode(enum.Enum):
    # Joints with rotational motions are expressed in degrees in nominal range of [-180, 180]
    DEGREE = 0
    # Joints with linear motions (like gripper of Aloha) are expressed in nominal range of [0, 100]
    LINEAR = 1


class JointOutOfRangeError(Exception):
    def __init__(self, message="Joint is out of range"):
        self.message = message
        super().__init__(self.message)


class StaraiMotorsBus:

    def __init__(
        self,
        config: StaraiMotorsBusConfig,
    ):
        self.port = config.port
        self.motors = config.motors
        self.mock = config.mock



        self.port_handler = None
        self.packet_handler = None
        self.calibration = None
        self.is_connected = False
        self.group_readers = {}
        self.group_writers = {}
        self.logs = {}
        self.gripper_degree_record = 0.0

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"StaraiMotorsBus({self.port}) is already connected. Do not call `motors_bus.connect()` twice."
            )

        if self.mock:
            # import tests.motors.mock_dynamixel_sdk as dxl
            raise RobotDeviceAlreadyConnectedError(
                f"mock is not supported."
            )


        self.uart = serial.Serial(port=self.port,baudrate=BAUDRATE,parity=serial.PARITY_NONE,stopbits=1,bytesize=8,timeout=0)
        motor_ids = []
        try:
            self.port_handler = uservo.UartServoManager(self.uart, srv_num=7,is_scan_servo = False)
            motor_names = self.motor_names

            for name in motor_names:
                motor_idx, model = self.motors[name]
                motor_ids.append(motor_idx)
                self.port_handler.reset_multi_turn_angle(motor_idx)
                time.sleep(0.01)

        except Exception:
            traceback.print_exc()
            print(
                "\nTry running `python lerobot/scripts/find_motors_bus_port.py` to make sure you are using the correct port.\n"
            )
            raise OSError(f"Failed to open port '{self.port}'.")
        time.sleep(1)
        self.port_handler.send_sync_servo_monitor(motor_ids)

        # Allow to read and write
        self.is_connected = True

    def reconnect(self):

        if self.mock:
            # import tests.motors.mock_dynamixel_sdk as dxl
            raise RobotDeviceAlreadyConnectedError(
                f"mock is not supported."
            )

        self.uart = serial.Serial(port=self.port,baudrate=BAUDRATE,parity=serial.PARITY_NONE,stopbits=1,bytesize=8,timeout=0.001)
        
        try:
            self.port_handler = uservo.UartServoManager(self.uart, srv_num=7)
        except Exception:
            raise OSError(f"Failed to open port '{self.port}'.")


        self.is_connected = True

    # def are_motors_configured(self):
    #     # Only check the motor indices and not baudrate, since if the motor baudrates are incorrect,
    #     # a ConnectionError will be raised anyway.
    #     try:
    #         return (self.motor_indices == self.read("ID")).all()
    #     except ConnectionError as e:
    #         print(e)
    #         return False

    def find_motor_indices(self, possible_ids=None, num_retry=2):
        if possible_ids is None:
            possible_ids = range(MAX_ID_RANGE)

        indices = []
        for idx in tqdm.tqdm(possible_ids):
            try:
                present_idx = self.read_with_motor_ids(self.motor_models, [idx], "ID", num_retry=num_retry)[0]
            except ConnectionError:
                continue

            if idx != present_idx:
                # sanity check
                raise OSError(
                    "Motor index used to communicate through the bus is not the same as the one present in the motor memory. The motor memory might be damaged."
                )
            indices.append(idx)

        return indices

    # def set_bus_baudrate(self, baudrate):
    #     present_bus_baudrate = self.port_handler.getBaudRate()
    #     if present_bus_baudrate != baudrate:
    #         print(f"Setting bus baud rate to {baudrate}. Previously {present_bus_baudrate}.")
    #         self.port_handler.setBaudRate(baudrate)

    #         if self.port_handler.getBaudRate() != baudrate:
    #             raise OSError("Failed to write bus baud rate.")

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def set_calibration(self, calibration: dict[str, list]):
        self.calibration = calibration

    def apply_calibration_autocorrect(self, values: np.ndarray | list, motor_names: list[str] | None):
        """This function applies the calibration, automatically detects out of range errors for motors values and attempts to correct.

        For more info, see docstring of `apply_calibration` and `autocorrect_calibration`.
        """
        try:
            values = self.apply_calibration(values, motor_names)
        except JointOutOfRangeError as e:
            print(e)
            # self.autocorrect_calibration(values, motor_names)
            values = self.apply_calibration(values, motor_names)
        return values

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        if motor_names is None:
            motor_names = self.motor_names
        values = values.astype(np.float32)

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                homing_offset = self.calibration["homing_offset"][calib_idx]
                _, model = self.motors[name]

                # Update direction of rotation of the motor to match between leader and follower.
                # In fact, the motor of the leader for a given joint can be assembled in an
                # opposite direction in term of rotation than the motor of the follower on the same joint.

                values[i] += homing_offset


                if (values[i] < LOWER_BOUND_DEGREE) or (values[i] > UPPER_BOUND_DEGREE):
                    raise JointOutOfRangeError(
                        f"Wrong motor position range detected for {name}. "
                        f"Expected to be in nominal range of [-{HALF_TURN_DEGREE}, {HALF_TURN_DEGREE}] degrees (a full rotation), "
                        f"with a maximum range of [{LOWER_BOUND_DEGREE}, {UPPER_BOUND_DEGREE}] degrees to account for joints that can rotate a bit more, "
                        f"but present value is {values[i]} degree. "
                        "This might be due to a cable connection issue creating an artificial 360 degrees jump in motor values. "
                        "You need to recalibrate by running: `python lerobot/scripts/control_robot.py calibrate`"
                    )

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Rescale the present position to a nominal range [0, 100] %,
                # useful for joints with linear motions like Aloha gripper
                values[i] = (values[i] - start_pos) / (end_pos - start_pos) * 100

                if (values[i] < LOWER_BOUND_LINEAR) or (values[i] > UPPER_BOUND_LINEAR):
                    raise JointOutOfRangeError(
                        f"Wrong motor position range detected for {name}. "
                        f"Expected to be in nominal range of [0, 100] % (a full linear translation), "
                        f"with a maximum range of [{LOWER_BOUND_LINEAR}, {UPPER_BOUND_LINEAR}] % to account for some imprecision during calibration, "
                        f"but present value is {values[i]} %. "
                        "This might be due to a cable connection issue creating an artificial jump in motor values. "
                        "You need to recalibrate by running: `python lerobot/scripts/control_robot.py calibrate`"
                    )

        return values

    # def autocorrect_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
    #     """This function automatically detects issues with values of motors after calibration, and correct for these issues.

    #     Some motors might have values outside of expected maximum bounds after calibration.
    #     For instance, for a joint in degree, its value can be outside [-270, 270] degrees, which is totally unexpected given
    #     a nominal range of [-180, 180] degrees, which represents half a turn to the left or right starting from zero position.

    #     Known issues:
    #     #1: Motor value randomly shifts of a full turn, caused by hardware/connection errors.
    #     #2: Motor internal homing offset is shifted by a full turn, caused by using default calibration (e.g Aloha).
    #     #3: motor internal homing offset is shifted by less or more than a full turn, caused by using default calibration
    #         or by human error during manual calibration.

    #     Issues #1 and #2 can be solved by shifting the calibration homing offset by a full turn.
    #     Issue #3 will be visually detected by user and potentially captured by the safety feature `max_relative_target`,
    #     that will slow down the motor, raise an error asking to recalibrate. Manual recalibrating will solve the issue.

    #     Note: A full turn corresponds to 360 degrees but also to 4096 steps for a motor resolution of 4096.
    #     """
    #     if motor_names is None:
    #         motor_names = self.motor_names

    #     # Convert from unsigned int32 original range [0, 2**32] to signed float32 range
    #     values = values.astype(np.float32)

    #     for i, name in enumerate(motor_names):
    #         calib_idx = self.calibration["motor_names"].index(name)
    #         calib_mode = self.calibration["calib_mode"][calib_idx]

    #         if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
    #             drive_mode = self.calibration["drive_mode"][calib_idx]
    #             homing_offset = self.calibration["homing_offset"][calib_idx]
    #             _, model = self.motors[name]
    #             resolution = self.model_resolution[model]

    #             # Update direction of rotation of the motor to match between leader and follower.
    #             # In fact, the motor of the leader for a given joint can be assembled in an
    #             # opposite direction in term of rotation than the motor of the follower on the same joint.
    #             if drive_mode:
    #                 values[i] *= -1

    #             # Convert from initial range to range [-180, 180] degrees
    #             calib_val = (values[i] + homing_offset) / (resolution // 2) * HALF_TURN_DEGREE
    #             in_range = (calib_val > LOWER_BOUND_DEGREE) and (calib_val < UPPER_BOUND_DEGREE)

    #             # Solve this inequality to find the factor to shift the range into [-180, 180] degrees
    #             # values[i] = (values[i] + homing_offset + resolution * factor) / (resolution // 2) * HALF_TURN_DEGREE
    #             # - HALF_TURN_DEGREE <= (values[i] + homing_offset + resolution * factor) / (resolution // 2) * HALF_TURN_DEGREE <= HALF_TURN_DEGREE
    #             # (- (resolution // 2) - values[i] - homing_offset) / resolution <= factor <= ((resolution // 2) - values[i] - homing_offset) / resolution
    #             low_factor = (-(resolution // 2) - values[i] - homing_offset) / resolution
    #             upp_factor = ((resolution // 2) - values[i] - homing_offset) / resolution

    #         elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
    #             start_pos = self.calibration["start_pos"][calib_idx]
    #             end_pos = self.calibration["end_pos"][calib_idx]

    #             # Convert from initial range to range [0, 100] in %
    #             calib_val = (values[i] - start_pos) / (end_pos - start_pos) * 100
    #             in_range = (calib_val > LOWER_BOUND_LINEAR) and (calib_val < UPPER_BOUND_LINEAR)

    #             # Solve this inequality to find the factor to shift the range into [0, 100] %
    #             # values[i] = (values[i] - start_pos + resolution * factor) / (end_pos + resolution * factor - start_pos - resolution * factor) * 100
    #             # values[i] = (values[i] - start_pos + resolution * factor) / (end_pos - start_pos) * 100
    #             # 0 <= (values[i] - start_pos + resolution * factor) / (end_pos - start_pos) * 100 <= 100
    #             # (start_pos - values[i]) / resolution <= factor <= (end_pos - values[i]) / resolution
    #             low_factor = (start_pos - values[i]) / resolution
    #             upp_factor = (end_pos - values[i]) / resolution

    #         if not in_range:
    #             # Get first integer between the two bounds
    #             if low_factor < upp_factor:
    #                 factor = math.ceil(low_factor)

    #                 if factor > upp_factor:
    #                     raise ValueError(f"No integer found between bounds [{low_factor=}, {upp_factor=}]")
    #             else:
    #                 factor = math.ceil(upp_factor)

    #                 if factor > low_factor:
    #                     raise ValueError(f"No integer found between bounds [{low_factor=}, {upp_factor=}]")

    #             if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
    #                 out_of_range_str = f"{LOWER_BOUND_DEGREE} < {calib_val} < {UPPER_BOUND_DEGREE} degrees"
    #                 in_range_str = f"{LOWER_BOUND_DEGREE} < {calib_val} < {UPPER_BOUND_DEGREE} degrees"
    #             elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
    #                 out_of_range_str = f"{LOWER_BOUND_LINEAR} < {calib_val} < {UPPER_BOUND_LINEAR} %"
    #                 in_range_str = f"{LOWER_BOUND_LINEAR} < {calib_val} < {UPPER_BOUND_LINEAR} %"

    #             logging.warning(
    #                 f"Auto-correct calibration of motor '{name}' by shifting value by {abs(factor)} full turns, "
    #                 f"from '{out_of_range_str}' to '{in_range_str}'."
    #             )

    #             # A full turn corresponds to 360 degrees but also to 4096 steps for a motor resolution of 4096.
    #             self.calibration["homing_offset"][calib_idx] += resolution * factor

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Inverse of `apply_calibration`."""
        if motor_names is None:
            motor_names = self.motor_names

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                homing_offset = self.calibration["homing_offset"][calib_idx]
                values[i] -= homing_offset
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]
                if i == 1:
                    if values[i] < start_pos:
                        values[i] = start_pos
                elif i == 2:
                    if values[i] > end_pos:
                        values[i] = end_pos
                    
            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Convert from nominal lnear range of [0, 100] % to
                # actual motor range of values which can be arbitrary.
                values[i] = values[i] / 100 * (end_pos - start_pos) + start_pos

        values = np.round(values).astype(np.int32)

        return values

    def read_with_motor_ids(self, motor_models, motor_ids, data_name, num_retry=NUM_READ_RETRY):
        if self.mock:
            import tests.motors.mock_dynamixel_sdk as dxl
        else:
            import dynamixel_sdk as dxl

        return_list = True
        if not isinstance(motor_ids, list):
            return_list = False
            motor_ids = [motor_ids]

        assert_same_address(self.model_ctrl_table, self.motor_models, data_name)
        addr, bytes = self.model_ctrl_table[motor_models[0]][data_name]
        group = dxl.GroupSyncRead(self.port_handler, self.packet_handler, addr, bytes)
        for idx in motor_ids:
            group.addParam(idx)

        for _ in range(num_retry):
            comm = group.txRxPacket()
            if comm == dxl.COMM_SUCCESS:
                break

        if comm != dxl.COMM_SUCCESS:
            raise ConnectionError(
                f"Read failed due to communication error on port {self.port_handler.port_name} for indices {motor_ids}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )

        values = []
        for idx in motor_ids:
            value = group.getData(idx, addr, bytes)
            values.append(value)

        if return_list:
            return values
        else:
            return values[0]

    def read(self, data_name, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"DynamixelMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()


        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        motor_ids = []
        models = []
        for name in motor_names:
            motor_idx, model = self.motors[name]
            motor_ids.append(motor_idx)
            models.append(model)

        # for _ in range(NUM_READ_RETRY):
        if data_name == "Present_Position":
            self.port_handler.send_sync_servo_monitor(motor_ids)
            comm = COMM_SUCCESS
            # break

        else:
            raise ConnectionError(
                f"function read not implemented for {data_name}"
            )  

        if comm != COMM_SUCCESS:
            raise ConnectionError(
                f"Read failed due to communication error on port {self.port}"
            )

        values = []
        for idx in motor_ids:
            values.append(self.port_handler.servos[idx].angle_monitor)
        # print(values[0],values[1],values[2],values[3],values[4],values[5],values[6])
        values = np.array(values)
        if data_name in CONVERT_UINT32_TO_INT32_REQUIRED:
            values = values.astype(np.int32)

        if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
            values = self.apply_calibration_autocorrect(values, motor_names)

        # log the number of seconds it took to read the data from the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "read", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # log the utc time at which the data was received
        ts_utc_name = get_log_name("timestamp_utc", "read", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

        return values

    def write_with_motor_ids(self, motor_models, motor_ids, data_name, values, num_retry=NUM_WRITE_RETRY):
        # if self.mock:
        #     import tests.motors.mock_dynamixel_sdk as dxl
        # else:
        #     import dynamixel_sdk as dxl

        if not isinstance(motor_ids, list):
            motor_ids = [motor_ids]
        if not isinstance(values, list):
            values = [values]

        # assert_same_address(self.model_ctrl_table, motor_models, data_name)
        # addr, bytes = self.model_ctrl_table[motor_models[0]][data_name]
        # group = dxl.GroupSyncWrite(self.port_handler, self.packet_handler, addr, bytes)
        # for idx, value in zip(motor_ids, values, strict=True):
        #     data = convert_to_bytes(value, bytes, self.mock)
        #     group.addParam(idx, data)

        # for _ in range(num_retry):
        #     comm = group.txPacket()
        #     if comm == dxl.COMM_SUCCESS:
        #         break

        if comm != dxl.COMM_SUCCESS:
            raise ConnectionError(
                f"Write failed due to communication error on port {self.port_handler.port_name} for indices {motor_ids}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"DynamixelMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        if isinstance(values, (int, float, np.integer)):
            values = [int(values)] * len(motor_names)

        values = np.array(values)
        # print(data_name,values)
        motor_ids = []
        models = []
        for name in motor_names:
            motor_idx, model = self.motors[name]
            motor_ids.append(motor_idx)
            models.append(model)

        if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
            values = self.revert_calibration(values, motor_names)

        values = values.tolist()
        if data_name == "Torque_Enable":
            comm = COMM_SUCCESS
        elif data_name == "Goal_Position":
            

            # command_data_list = [struct.pack('<BhHH',motor_ids[i],int(values[i]*10), 200, 0)for i in motor_ids]
            # self.port_handler.send_sync_angle(self.port_handler.CODE_SET_SERVO_ANGLE,len(motor_ids),command_data_list)

            if  motor_names[6] != None and motor_names[6] == "gripper":
                if self.gripper_degree_record != values[6] :
                    self.gripper_degree_record = values[6]
                    command_data_list = [struct.pack("<BlLHHH", motor_ids[i], int(values[i]*10), 100, 50, 50, 0)for i in motor_ids]
                    self.port_handler.send_sync_multiturnanglebyinterval(self.port_handler.CODE_SET_SERVO_ANGLE_MTURN_BY_INTERVAL,len(motor_ids), command_data_list)
                else:
                    command_data_list = [struct.pack("<BlLHHH", motor_ids[i], int(values[i]*10), 100, 50, 50, 0)for i in (motor_ids[:-1])]
                    self.port_handler.send_sync_multiturnanglebyinterval(self.port_handler.CODE_SET_SERVO_ANGLE_MTURN_BY_INTERVAL,len(motor_ids[:-1]), command_data_list)
            else:
                command_data_list = [struct.pack("<BlLHHH", motor_ids[i], int(values[i]*10), 100, 50, 50, 0)for i in motor_ids]
                self.port_handler.send_sync_multiturnanglebyinterval(self.port_handler.CODE_SET_SERVO_ANGLE_MTURN_BY_INTERVAL,len(motor_ids), command_data_list)
            comm = COMM_SUCCESS
 



        else :
            raise ValueError(
                f"Write failed for data_name {data_name} because it is not supported. "
            )





        if comm != COMM_SUCCESS:
            raise ConnectionError(
                f"Write failed due to communication error on port {self.port} for data_name {data_name}: "
                # f"{self.packet_handler.getTxRxResult(comm)}"
            )

        # log the number of seconds it took to write the data to the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "write", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # TODO(rcadene): should we log the time before sending the write command?
        # log the utc time when the write has been completed
        ts_utc_name = get_log_name("timestamp_utc", "write", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"DynamixelMotorsBus({self.port}) is not connected. Try running `motors_bus.connect()` first."
            )

        if self.port_handler is not None:
            self.port_handler = None

        self.packet_handler = None
        self.group_readers = {}
        self.group_writers = {}
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
