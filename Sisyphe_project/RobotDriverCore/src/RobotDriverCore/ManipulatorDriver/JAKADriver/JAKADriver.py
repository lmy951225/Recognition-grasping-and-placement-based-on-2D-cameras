from typing import List, Union, Dict, Tuple, cast
import os
import copy
import math
import json
import threading
import socket
import logging
import time
import numpy as np
from RobotDriverCore.ManipulatorDriver.ManipulatorDriver import ManipulatorDriver
from RobotDriverCore.RobotDriverCoreUtils.ResponseStatus import ResponseStatus
from RobotDriverCore.RobotDriverCoreUtils.RobotConstraints import RobotConstraints
from RobotDriverCore.RobotDriverCoreUtils.RobotPosition import CartesianPosition, JointPosition
from RobotDriverCore.RobotDriverCoreUtils.SignalIO import SignalIOType
from RobotDriverCore.TrajectoryPlanner.FileManager import FileManager
from RobotDriverCore.TrajectoryPlanner.Trajectory.JsonTrajectory import JsonTrajectory


class JAKARobotState():
    def __init__(self, data: dict) -> None:
        self.__raw_data = data

    def __deepcopy__(self, memo) -> 'JAKARobotState':
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
    
    @property
    def raw_data(self)->Dict:
        return self.__raw_data

    @property
    def data_len(self) -> int:
        return self.__raw_data['len']
    
    @property
    def is_drag_status(self) -> bool:
        return self.__raw_data['drag_status']

    @property
    def error_data(self) -> Tuple[str, str]:
        return self.__raw_data['errcode'], self.__raw_data['errmsg']
 
running_state_thread_flag = False
class JAKAsocketTCP():
    def __init__(self, logger: logging.Logger) -> None:
        self.__commander = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__robot_state = None
        self.__robot_state_recorder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__get_robot_thread = threading.Thread(target=self._get_robot_state)
        self.__logger = logger
    
    @property
    def robot_state(self) -> JAKARobotState:
        return copy.deepcopy(self.__robot_state)

    def _get_robot_state(self):
        self.__robot_state_recorder.settimeout(0.5)
        while (1):
            try:
                data = json.loads(self.__robot_state_recorder.recv(16384).decode())
                if data:
                    self.__robot_state = JAKARobotState(data=data)
            except Exception as e:
                print(e)
            if not running_state_thread_flag:
                break
        
    def connect(self, ip_address: str) -> bool:
        commander_connect_code = self.__commander.connect_ex((ip_address, 10001))
        state_connect_code = self.__robot_state_recorder.connect_ex((ip_address, 10000))
        if state_connect_code == 0:
            running_state_thread_flag = True
            self.__get_robot_thread.start()
        return (commander_connect_code == 0) and (state_connect_code == 0)
    
    def disconnect(self):
        running_state_thread_flag = False
        self.__commander.close()
        self.__robot_state_recorder.close()
    

    def do_JAKA_command(self, command_info: dict, print_logger: bool = True) -> Union[str, dict]:
        result = 'socket send data error'
        try:
            self.__commander.sendall(json.dumps(command_info).encode())
            result = json.loads(self.__commander.recv(2048).decode())
            if print_logger:
                self.__logger.info("JAKA SEND DATA: {}".format(command_info))
                self.__logger.info("JAKA RESPONSE DATA: {}".format(result))
            if result['cmdName'] != command_info['cmdName']:
                return "sender and receiver is not same result"
        except:
            pass
        return result

###########################JAKA helper functions######################
def generateJAKAcmd(command: str) -> Dict:
    return {"cmdName":command}

class JAKAresponse():
    def __init__(self, receive_data: Dict) -> None:
        self.__receive_data = receive_data
        self.__error_code = int(receive_data['errorCode'])
        self.__error_msg = receive_data['errorMsg']
        self.__command_name = receive_data['cmdName']

    def __getitem__(self, key):
        if key in self.__receive_data:
            return self.__receive_data[key]
        else:
            raise KeyError("no key {} included in response".format(key))

    @property
    def is_ok(self) -> bool:
        return self.__error_code == 0

    @property
    def error_code(self) -> int:
        return self.__error_code
    
    @property
    def error_msg(self) -> str:
        return self.__error_msg
    
    @property
    def command_name(self) -> str:
        return self.__command_name
    
    def to_ResponseStatus(self) -> ResponseStatus:
        if self.__error_code == 0:
            return ResponseStatus.OK
        else:
            return ResponseStatus.OK.str2ResponseStatus(self.__error_msg)
    
class JAKADriver(ManipulatorDriver):
    def __init__(self) -> None:
        super().__init__()
        self.__s = JAKAsocketTCP(self._logger)

    @property
    def JAKA_robot_state(self) -> JAKARobotState:
        return self.__s.robot_state

    def __general_JAKA_command(self, cmd_name: Union[str, Dict[str, Union[float, str, List]]], all_response: bool = True) -> ResponseStatus:
        """ do JAKA command and return ResponseStatus, only work for interfaces that care about 

        Args:
            cmd_name (Union[str, Dict[str, Union[float, str, List]]]): JAKA cmd name or JAKA cmd dict
            all_response (bool) : return all response or not
        Returns:
            ResponseStatus: command response
        """
        cmd_json = generateJAKAcmd(cmd_name) if isinstance(cmd_name, str) else cmd_name
        result = self.__s.do_JAKA_command(cmd_json)
        if isinstance(result, str):
            return ResponseStatus.OK.str2ResponseStatus(result)
        else:
            jaka_response = JAKAresponse(result)
            if all_response or not jaka_response.is_ok:
                return jaka_response.to_ResponseStatus()
            return jaka_response
        
    def power_on(self) -> ResponseStatus:
        return self.__general_JAKA_command(cmd_name='power_on') 

    def power_off(self) -> ResponseStatus:
        return self.__general_JAKA_command(cmd_name='power_off')

    def connect(self, controller_ip: str) -> ResponseStatus:
        self._robot_ip = controller_ip
        if self.__s.connect(ip_address=controller_ip):
            self.__s.do_JAKA_command(generateJAKAcmd('get_version'))
            return ResponseStatus.OK
        else:
            return ResponseStatus.CONNECTION_TIMEOUT
        
    def disconnect(self) -> ResponseStatus:
        self.__s.disconnect()
        return ResponseStatus.OK
        
    def is_connect(self) -> ResponseStatus:
        return self.__general_JAKA_command(cmd_name='get_robot_state')
    
    def reset(self) -> ResponseStatus:
        return self.__general_JAKA_command(cmd_name="clear_error")
    
    def forward_kinematics(self, joint_position: JointPosition) -> Union[ResponseStatus, CartesianPosition]:
        fk_cmd = generateJAKAcmd('kine_forward')
        fk_cmd['jointPosition'] = [joint_position.j1, joint_position.j2, joint_position.j3, joint_position.j4,joint_position.j5, joint_position.j6]
        forward_info_jaka = self.__general_JAKA_command(fk_cmd, all_response=False)
        if isinstance(forward_info_jaka, ResponseStatus):
            return forward_info_jaka
        return CartesianPosition(xyzrxryrz=forward_info_jaka['cartPosition'])
    
    def inverse_kinematics(self, cartesian_position: CartesianPosition, reference_joint_position: JointPosition) -> Union[ResponseStatus, JointPosition]:
        ik_cmd = generateJAKAcmd("kine_inverse")
        ik_cmd['cartPosition'] = [cartesian_position.x, cartesian_position.y, cartesian_position.z, cartesian_position.rx, cartesian_position.ry, cartesian_position.rz] 
        ik_cmd['jointPosition'] = [reference_joint_position.j1, reference_joint_position.j2, reference_joint_position.j3, reference_joint_position.j4, reference_joint_position.j5, reference_joint_position.j6]
        inverse_info_jaka = self.__general_JAKA_command(ik_cmd, all_response=False)
        if isinstance(inverse_info_jaka, ResponseStatus):
            return inverse_info_jaka
        return JointPosition(joints=inverse_info_jaka['jointPosition'])
    
    def get_name(self) -> Union[str, ResponseStatus]:
        return super().get_name()
        
    def enable(self) -> ResponseStatus:
        return self.__general_JAKA_command(cmd_name='enable_robot') 
    
    def disable(self) -> ResponseStatus:
        return self.__general_JAKA_command(cmd_name="disable_robot")
    
    def set_speed_ratio(self, speed_ratio: float) -> ResponseStatus:
        self._speed_ratio = speed_ratio
        speed_cmd = generateJAKAcmd(command='rapid_rate')
        speed_cmd['rate_value'] = speed_ratio
        return self.__general_JAKA_command(speed_cmd)
    
    def get_io(self, signal_type: SignalIOType, ports: List[int]) -> Union[List[bool], ResponseStatus]:
        """get io from JAKA cobot

        Args:
            signal_type (SignalIOType): io singal type, only support DO right now
            ports (List[int]): io ports to read

        Returns:
            Union[List[bool], ResponseStatus]: _description_
        """
        if signal_type == SignalIOType.SIGNAL_DI:
            io_info_jaka = self.__general_JAKA_command(cmd_name='get_din_status', all_response=False) 
            if isinstance(io_info_jaka, ResponseStatus):
                return io_info_jaka
            din_status = io_info_jaka['din_status']
            results: List[bool] = list()
            for port in ports:
                results.append(din_status[port])

        return results
    
    def set_io(self, signal_type: SignalIOType, ports: List[int], io_value: List[bool], JAKA_io_type: int = 0) -> ResponseStatus:
        """set io for JAKA cobot

        Args:
            signal_type (SignalIOType): io signal type, only support DO right now
            ports (List[int]): io ports to set
            io_value (List[bool]): io value to set
            JAKA_io_type (int, optional): JAKA io type, 0 represent controller, 1 represent tcp, 2 represent extended io. Defaults to 0.

        Returns:
            ResponseStatus: set result
        """
        if signal_type == SignalIOType.SIGNAL_DO:
            set_io_command = generateJAKAcmd('set_digital_output')
            for port, value in zip(ports, io_value):
                set_io_command['type'] = JAKA_io_type
                set_io_command['index'] = port
                set_io_command['value'] = int(value)
                set_io_result = self.__general_JAKA_command(cmd_name=set_io_command)
                if set_io_result != ResponseStatus.OK:
                    return set_io_result
        else:
            return ResponseStatus.UNSUPPORTED_SIGNAL_TYPE
        return ResponseStatus.OK

    
    def get_DH_parameters(self) -> Union[List, ResponseStatus]:
        dh_info_jaka = self.__general_JAKA_command(cmd_name='get_dh_param', all_response=False) 
        if isinstance(dh_info_jaka, ResponseStatus):
            return dh_info_jaka
        dh_parameters: List[List[float]] = list()
        for a, alpha, d, offset in zip(dh_info_jaka['dh_param'][1], dh_info_jaka['dh_param'][0],dh_info_jaka['dh_param'][2],dh_info_jaka['dh_param'][3]):
            dh_parameters.append([a / 1000, math.radians(alpha), d / 1000, math.radians(offset)])
        return dh_parameters
    
    def get_current_joint_position(self) -> Union[JointPosition, ResponseStatus]:
        joint_info_jaka = self.__general_JAKA_command(cmd_name='get_joint_pos', all_response=False) 
        if isinstance(joint_info_jaka, ResponseStatus):
            return joint_info_jaka
        return JointPosition(joints=joint_info_jaka['joint_pos'])
    
    def move2joint_position_async(self, joint_position: Union[JointPosition, CartesianPosition], jaka_move_parameters: Dict[str, float] = dict()) -> ResponseStatus:
        if isinstance(joint_position, CartesianPosition):
            current_joint_position = self.get_current_joint_position()
            if isinstance(current_joint_position, ResponseStatus):
                return current_joint_position
            joint_position = self.inverse_kinematics(cartesian_position=joint_position, reference_joint_position=current_joint_position)
            if isinstance(joint_position, ResponseStatus):
                return joint_position
        move_joint_command = generateJAKAcmd('joint_move')
        move_joint_command["speed"] = jaka_move_parameters['speed'] if jaka_move_parameters.get("speed", False) else 20.5
        move_joint_command["accel"] = jaka_move_parameters['accel'] if jaka_move_parameters.get("accel", False) else 20.5
        move_joint_command['relFlag'] = 0
        move_joint_command['jointPosition'] = [joint_position.j1, joint_position.j2, joint_position.j3, joint_position.j4, joint_position.j5, joint_position.j6]
        return self.__general_JAKA_command(cmd_name=move_joint_command)
    
    def move2joint_position_sync(self, joint_position: Union[JointPosition, CartesianPosition], jaka_move_parameters: Dict[str, float] = dict()) -> ResponseStatus:
        move_joint_response = self.move2joint_position_async(joint_position, jaka_move_parameters)
        if move_joint_response != ResponseStatus.OK:
            return move_joint_response
        total_wait_time = jaka_move_parameters['total_wait_time'] if jaka_move_parameters.get("total_wait_time", False) else 20
        wait_time_step = jaka_move_parameters['wait_time_step'] if jaka_move_parameters.get("wait_time_step", False) else 1
        return self.wait_until_manipulator_ready(total_wait_time=total_wait_time / self._speed_ratio, wait_time_step=wait_time_step)
        
    def move_line_async(self, position: Union[JointPosition, CartesianPosition], jaka_move_parameters: Dict[str, float] = dict()) -> ResponseStatus:
        if not isinstance(position, CartesianPosition):
            return ResponseStatus.UNSUPPORTED_PARAMETER
        cartesian_position = cast(CartesianPosition, position)
        move_line_command = generateJAKAcmd('moveL')
        move_line_command["speed"] = jaka_move_parameters['speed'] if jaka_move_parameters.get("speed", False) else 20.5
        move_line_command["accel"] = jaka_move_parameters['accel'] if jaka_move_parameters.get("accel", False) else 50
        move_line_command["tol"] = jaka_move_parameters['tol'] if jaka_move_parameters.get("tol", False) else 0.5
        move_line_command['relFlag'] = 0
        move_line_command['cartPosition'] = [cartesian_position.x, cartesian_position.y, cartesian_position.z, cartesian_position.rx, cartesian_position.ry, cartesian_position.rz]
        return self.__general_JAKA_command(cmd_name=move_line_command)
    
    def move_line_sync(self, position: Union[JointPosition, CartesianPosition], jaka_move_parameters: Dict[str, float] = dict()) -> ResponseStatus:
        move_line_response = self.move_line_async(position, jaka_move_parameters)
        if move_line_response != ResponseStatus.OK:
            return move_line_response
        total_wait_time = jaka_move_parameters['total_wait_time'] if jaka_move_parameters.get("total_wait_time", False) else 20
        wait_time_step = jaka_move_parameters['wait_time_step'] if jaka_move_parameters.get("wait_time_step", False) else 1
        return self.wait_until_manipulator_ready(total_wait_time=total_wait_time / self._speed_ratio, wait_time_step=wait_time_step)
    
    def get_current_cartesian_position(self) -> Union[JointPosition, ResponseStatus]:
        tcp_info_jaka = self.__general_JAKA_command(cmd_name='get_tcp_pos', all_response=False) 
        if isinstance(tcp_info_jaka, ResponseStatus):
            return tcp_info_jaka
        return CartesianPosition( xyzrxryrz=tcp_info_jaka['tcp_pos'])
    
    def wait_until_manipulator_ready(self, total_wait_time: float, wait_time_step: float) -> ResponseStatus:
        start = time.time()
        count_joint = 0
        last_joint_position = self.get_current_joint_position()
        if isinstance(last_joint_position, ResponseStatus):
            return last_joint_position
        while True:
            time.sleep(wait_time_step)
            if time.time() - start > total_wait_time:
                return ResponseStatus.WAITING_TIMEOUT
            current_joint_position = self.get_current_joint_position()
            count_joint += 1
            if isinstance(current_joint_position, ResponseStatus):
                return current_joint_position
            if last_joint_position.is_nearly_equal(current_joint_position):
                if count_joint == 1:
                    return ResponseStatus.UNSUPPORTED_PARAMETER
                protective_stop_info = self.__general_JAKA_command(cmd_name="protective_stop_status", all_response=False)
                if isinstance(protective_stop_info, ResponseStatus):
                    return protective_stop_info
                else:
                    if protective_stop_info['protective_stop'] == '0':
                        return ResponseStatus.OK
                    else:
                        return ResponseStatus.OK.str2ResponseStatus("protective stop state")
            last_joint_position = current_joint_position
    
    def csv2executable_trajectory(self, trajectory_dir: str, trajectory_name: str) -> ResponseStatus:
        return super().csv2executable_trajectory(trajectory_dir, trajectory_name)
    
    def execute_trajectory(self, trajectory_dir: str, trajectory_name: str) -> ResponseStatus:
        return super().execute_trajectory(trajectory_dir, trajectory_name)
    
    def read_program_info(self, program_name: str) -> ResponseStatus | Tuple[List[JointPosition], List[CartesianPosition]]:
        return super().read_program_info(program_name)
    
    def run_program_async(self, program_name: str) -> ResponseStatus:
        # load program
        load_program_cmd = generateJAKAcmd(command='load_program')
        load_program_cmd['programName'] = "track/" + program_name + '.jks'
        load_status = self.__general_JAKA_command(load_program_cmd)
        if load_status != ResponseStatus.OK:
            return load_status
        # run program
        return self.__general_JAKA_command(generateJAKAcmd("play_program"))

    def stop_program(self, program_name: str) -> ResponseStatus:
        "only stop current running program"
        return self.__general_JAKA_command(generateJAKAcmd("stop_program"))
    
    def wait_until_program_finished(self, program_name: str, total_wait_time: float, wait_time_step: float) -> ResponseStatus:
        start = time.time()
        get_program_state_cmd = generateJAKAcmd("get_program_state")
        while (True):
            program_state_info = self.__s.do_JAKA_command(get_program_state_cmd)
            if isinstance(program_state_info, str):
                return ResponseStatus.OK.str2ResponseStatus(program_state_info)
            program_state_jaka = JAKAresponse(program_state_info)
            if not program_state_jaka.is_ok:
                return program_state_jaka.to_ResponseStatus()
            if program_state_jaka['pogramState'] == 'running':
                time.sleep(wait_time_step)
            if time.time() - start > total_wait_time:
                return ResponseStatus.WAITING_TIMEOUT