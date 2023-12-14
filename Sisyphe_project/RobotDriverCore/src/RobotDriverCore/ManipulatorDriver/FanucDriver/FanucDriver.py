from typing import List, Union
import os
import socket
import logging
import time
import pandas
import numpy as np
from aenum import extend_enum
from RobotDriverCore.ManipulatorDriver.ManipulatorDriver import ManipulatorDriver
from RobotDriverCore.RobotDriverCoreUtils.ResponseStatus import ResponseStatus
from RobotDriverCore.RobotDriverCoreUtils.RobotConstraints import RobotConstraints
from RobotDriverCore.RobotDriverCoreUtils.RobotPosition import CartesianPosition, JointPosition
from RobotDriverCore.RobotDriverCoreUtils.SignalIO import SignalIOType
from RobotDriverCore.TrajectoryPlanner.FileManager import FileManager
from RobotDriverCore.TrajectoryPlanner.Trajectory.JsonTrajectory import JsonTrajectory
from RobotDriverCore.ManipulatorDriver.FanucDriver.FanucStreamMotionHelper import FanucJ519_stream_motion_execute, FanucJ519_stream_motion_prepare

class SocketTCPIP():
    def __init__(self, tcp_port: int, logger: logging.Logger) -> None:
        self.__tcp_port = tcp_port
        self.__logger = logger
        self.__s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__s.settimeout(1.2) # Karel RESET command requests at least over 1 second 

    def connect(self, ip_address: str) -> bool:
        connect_code = self.__s.connect_ex((ip_address, self.__tcp_port))
        return connect_code == 0

    def disconnect(self):
        self.__s.close()

    def send_message(self, data: bytes, print_logger: bool = True) -> bytes:
        response_data = b''
        try:
            self.__s.send(data)
            response_data = self.__s.recv(256)
            if print_logger:
                self.__logger.info("FANUC SEND DATA: {}".format(data))
                self.__logger.info("FANUC RESPONSE DATA: {}".format(response_data))
        except:
            pass
        return response_data

# general command 
CHECK_CONNECTION = '01'
CHECK_DISCONNECTION = '02' # meanless, miscalculation in design
RESET = '03' 
GET_ORDER_NUM = '04' # map order number to fanuc robot seriers
SET_SPEED = '07' # set speed command, from 10~100
READ_IO = '08' # read io information
WRITE_IO = '09' # write io information
GET_JOINT_POSE = '10' # read current joint position (J1~J6)
GET_CARTESIAN_POSE = '11' # read current cartesian position (X Y Z RX RY RZ)
# motion command
ENABLE_FANUC = '05' # soft swith, not implementation yet
DISABLE_FANUC = '06' # soft swith, not implementation yet
MOVE_JOINT = '12' # move the robot to target joint position
START_J519 = '13' # start the TP program for Fanuc 
END_J519 = '14' # end the TP program for Fanuc
START_TASK = '15' # run task with name
STOP_TASK = '16' # abort task with name
CHECK_TASK_DONE = '17' # check if the task is done
# response
COMMAND_OK = '0'
COMMAND_OK_BYTE = b'0'
COMMAND_FAIL = '1'
COMMAND_FAIL_BYTE = b'1'

# state server
TP_ENABLE = '03' # check if TP is enable
TP_EMERGENCY_STOP = '04' # check if EMERGENCY_STOP is pressed
ROBOT_MOVE = '05' # check if robot is move
ROBOT_FAULT = '06' # check if robot has fault
GET_ACTIVE_ALARM = '99' # read active alarm
STATE_YES = '1'
STATE_YES_BYTE = b'1'
STATE_NO = '0'
STATE_NO_BYTE = b'0'

class FanucDriver(ManipulatorDriver):
    def __init__(self, is_ROBOTGUIDE: bool = True) -> None:
        super().__init__()
        self.__is_ROBOTGUIDE = is_ROBOTGUIDE
        self.__command_client_s3 = SocketTCPIP(10010, self._logger)
        self.__motion_client_s4 = SocketTCPIP(10011, self._logger)
        self.__state_client_s5 = SocketTCPIP(10012, self._logger)
        self.__controller_ip = str()
    
    def _generate_robot_constraints(self) -> RobotConstraints:
        """Fanuc constraints is from https://www.shanghai-fanuc.com.cn/uploadfile/3049a71ee39d4e759f332bf0c5f09f00.pdf and stream motion protocol(J519)

        Returns:
            RobotConstraints: min constraints of LR Mate 200iD and LR Mate 200iD/7L
        """
        robot_constraint = RobotConstraints() # unit is radian
        robot_constraint.time_step = 0.008
        # different limitS for                       J1     J2      J3      J4      J5      J6
        robot_constraint.max_position       =np.asarray([2.9,  2.5,    4.8,    3.3,    2.1,   6.2])
        robot_constraint.min_position       =np.asarray([-2.9, -1.7,   -2.4,   -3.3,   -2.1,    -6.2])
        robot_constraint.velocity_limit     =np.asarray([5.16, 4.32,   5.72,   7.6,    7.6,    13.96])
        robot_constraint.acceleration_limit =np.asarray([21.52,18.03,  23.85,  31.99,  31.7,   58.17])
        robot_constraint.jerk_limit         =np.asarray([179,  150,    198,    266,    264,    484])
        return robot_constraint

    def connect(self, controller_ip: str) -> ResponseStatus:
        if self.__is_ROBOTGUIDE:
            controller_ip = "127.0.0.1"
        self.__controller_ip = controller_ip
        if self.__command_client_s3.connect(controller_ip) and self.__motion_client_s4.connect(controller_ip) and self.__state_client_s5.connect(controller_ip):
            return ResponseStatus.OK
        else:
            return ResponseStatus.CONNECTION_TIMEOUT
        
    def disconnect(self) -> ResponseStatus:
        self.__command_client_s3.disconnect()
        self.__motion_client_s4.disconnect()
        self.__state_client_s5.disconnect()
        return ResponseStatus.OK
    
    def is_connect(self) -> ResponseStatus:
        """check connection of 3 servers

        Returns:
            ResponseStatus: connection results
        """
        connect_command = CHECK_CONNECTION.encode()
        if self.__command_client_s3.send_message(connect_command, print_logger=False) == COMMAND_OK_BYTE and \
            self.__motion_client_s4.send_message(connect_command, print_logger=False) == COMMAND_OK_BYTE and \
            self.__state_client_s5.send_message(connect_command, print_logger=False) == COMMAND_OK_BYTE:
            return ResponseStatus.OK
        else:
            return ResponseStatus.CONNECTION_TIMEOUT
        
    def reset(self) -> ResponseStatus:
        return ResponseStatus.OK if self.__command_client_s3.send_message(RESET.encode()) == COMMAND_OK_BYTE else ResponseStatus.CONTROLLER_ERROR
    
    def get_name(self) -> Union[str, ResponseStatus]:
        name_result = self.__command_client_s3.send_message(GET_ORDER_NUM.encode()).decode('utf-8')
        if name_result and name_result[0] == COMMAND_OK:
            name_str = name_result[1:]
            if "755" in name_str:
                return "FANUC LR Mate 200iD"
            elif "750" in name_str:
                return "FANUC LR Mate 200iD/7L"
            else:
                return "unknow Fanuc type {}".format(name_str)
        return ResponseStatus.CONNECTION_TIMEOUT
    
    def enable(self) -> ResponseStatus:
        self._soft_enable = True
        return ResponseStatus.OK 

    def disable(self) -> ResponseStatus:
        self._soft_enable = False
        return ResponseStatus.OK 
    
    def set_speed_ratio(self, speed_ratio: float) -> ResponseStatus:
        """set speed for fanuc. 
            COMMAND FORMAT:  1~2 bytes: set speed command; 3~5 bytes: speed bytes, '010'~'100'
            RETURN FORMAT: 1 byte, ok or fail
        Args:
            speed_ratio (float): only support 0 < speed_ratio <= 1

        Returns:
            ResponseStatus: _description_
        """
        self._speed_ratio = speed_ratio
        speed = round(100 * speed_ratio)
        set_speed_command:str = SET_SPEED
        if speed >= 100:
            set_speed_command += '100'
        elif speed <= 10:
            set_speed_command += '010'
        else:
            set_speed_command += '0' + str(speed)
        return ResponseStatus.OK if self.__command_client_s3.send_message(set_speed_command.encode()) == COMMAND_OK_BYTE else ResponseStatus.CONTROLLER_ERROR 

    def get_io(self, signal_type: SignalIOType, ports: List[int]) -> Union[List[bool], ResponseStatus]:
        """read io by signal type and ports
            COMMAND FORMAT: 1~2 bytes: read io command; 3~4 bytes, fanuc signal type same with J519 defined; rest is like [port1, port2, port3......port n-1, port n], seperate by ','
            RETURN FORMAT: 1st byte: ok or fail ;io values by ports index, seperate by ',', for example '1,0,0,0,1,1,0'
        Args:
            signal_type (SignalIOType): singal type to read
            ports (List[int]): ports number to read

        Returns:
            Union[List[bool], ResponseStatus]: io information or fail result
        """
        type_code = signal_type.Fanuc_code
        if type_code <= 0 or type_code >=36:
            return ResponseStatus.UNSUPPORTED_SIGNAL_TYPE
        command_str: str = READ_IO
        type_code_str = str(type_code)
        if len(type_code_str) == 1:
            type_code_str = '0' + type_code_str
        command_str += type_code_str
        for port in ports:
            command_str += ','
            command_str += str(port)
        response_data = self.__command_client_s3.send_message(command_str.encode()).decode()
        if response_data and response_data[0] == COMMAND_OK:
            results: List[bool] = list()
            io_str_list = response_data[1:].split(',')
            for tmp_str in io_str_list:
                if tmp_str == '1':
                    results.append(True)
                elif tmp_str == '0':
                    results.append(False)
            return results
        return ResponseStatus.CONTROLLER_ERROR
    
    def set_io(self, signal_type: SignalIOType, ports: List[int], io_value: List[bool]) -> ResponseStatus:
        """write io information
            COMMAND FORMAT: 1~2 bytes: read io command; 3~4 bytes, fanuc signal type same with J519 defined; rest is like [port1 1/0, port2 1/0, ......, port n 1/0]
            RETURN FORMAT: 1st byte: ok or fail 
        Args:
            signal_type (SignalIOType): single type to write
            ports (List[int]): port index list
            io_value (List[bool]): write io values

        Returns:
            ResponseStatus: write io result
        """
        type_code = signal_type.Fanuc_code
        if type_code <= 0 or type_code >=36:
            return ResponseStatus.UNSUPPORTED_SIGNAL_TYPE
        command_str: str = WRITE_IO
        type_code_str = str(type_code)
        if len(type_code_str) == 1:
            command_str += '0'
        command_str += type_code_str
        for port, io_flag in zip(ports, io_value):
            command_str += ','
            command_str += str(port)
            if io_flag:
                command_str += '1'
            else:
                command_str += '0'
        return ResponseStatus.OK if self.__command_client_s3.send_message(command_str.encode()) == COMMAND_OK_BYTE else ResponseStatus.CONTROLLER_ERROR

    def get_current_joint_position(self) -> Union[JointPosition, ResponseStatus]:
        """get joint position of fanuc robot
            COMMAND FORMAT: 1~2 bytes: get joint position command
            RETURN FORMAT: 1st byte: ok or fail; rest bytes (if ok): j1, j2, j3, j4, j5, j6
        Returns:
            Union[JointPosition, ResponseStatus]: joint position (OK) or failure response
        """
        joint_position_results = self.__command_client_s3.send_message(GET_JOINT_POSE.encode(), print_logger=False).decode()
        if joint_position_results and joint_position_results[0] == COMMAND_OK:
            joint_str_list = [joint_str for joint_str in joint_position_results[1:].split(',') if joint_str]
            if len(joint_str_list) >= 6:
                joints = [float(s) for s in joint_str_list]
                return JointPosition(joints=joints) 
        return ResponseStatus.CONTROLLER_ERROR
    
    def get_current_cartesian_position(self) -> Union[CartesianPosition, ResponseStatus]:
        """get cartesian of fanuc robot
            COMMAND FORMAT: 1~2 bytes: get cartesian position command
            RETURN FORMAT: 1st byte: ok or failt; rest bytes(if ok): x,y,z,rx,ry,rz
        Returns:
            Union[CartesianPosition, ResponseStatus]: cartesian position (OK) or failure
        """
        cartesian_position_results = self.__command_client_s3.send_message(GET_CARTESIAN_POSE.encode(), print_logger=False).decode()
        if cartesian_position_results and cartesian_position_results[0] == COMMAND_OK:
            cartesian_str_list = [catesian_value for catesian_value in cartesian_position_results[1:].split(',') if catesian_value]
            if len(cartesian_str_list) == 6:
                xyzrxryrz = [float(s) for s in cartesian_str_list]
                xyzrxryrz[2] = xyzrxryrz[2] + 330
                return CartesianPosition(xyzrxryrz=xyzrxryrz)
        return ResponseStatus.CONTROLLER_ERROR
    
    def wait_until_manipulator_ready(self, total_wait_time: float, wait_time_step: float) -> ResponseStatus:
        """check the manipulator states until not moving, 
            then check MoveJ and flyshot TP programs 
            last check current active alarm

        Args:
            total_wait_time (float): total wait time 
            wait_time_step (float): check state interval

        Returns:
            ResponseStatus: response for waiting process
        """
        start = time.time()
        # check robot move
        while True:
            if self.__state_client_s5.send_message(ROBOT_MOVE.encode()) != STATE_YES_BYTE:
                break
            time.sleep(wait_time_step)
            if time.time() - start >= total_wait_time:
                return ResponseStatus.WAITING_TIMEOUT
        # check flyshot program
        flyshot_program_response = self.wait_until_program_finished(program_name='WEIYI_FLYSHOT', total_wait_time=5, wait_time_step=0.1)
        if flyshot_program_response != ResponseStatus.OK:
            return flyshot_program_response
        # check move j program
        move_joint_program_response = self.wait_until_program_finished(program_name='WEIYI_MOVEJ', total_wait_time=5, wait_time_step=0.1)
        if move_joint_program_response != ResponseStatus.OK:
            return move_joint_program_response
        # check alarm results
        alarm_results = self.__state_client_s5.send_message(GET_ACTIVE_ALARM.encode()).decode('gbk')
        if alarm_results:
            if alarm_results == STATE_NO:
                return ResponseStatus.OK
            else:
                if alarm_results not in ResponseStatus.__dict__:
                    extend_enum(ResponseStatus, alarm_results, (999, alarm_results))
                return ResponseStatus[alarm_results]
        else:
            return ResponseStatus.CONTROLLER_ERROR

    def move2joint_position_async(self, joint_position: JointPosition) -> ResponseStatus:
        """move fanuc manipulator to target position asynchronously
            COMMAND FORMAT: 1~2 bytes: move joint command; rest bytes [j1,j2,j3,j4,j5,j6]
            RETURN FORMAT: 1st byte: ok or fail
        Args:
            joint_position (JointPosition): target joint position

        Returns:
            ResponseStatus: response only represent the move_j command response, not responsible for final condition
        """
        move_joint_cmd_str = MOVE_JOINT
        move_joint_cmd_str += str(joint_position.j1) + ','
        move_joint_cmd_str += str(joint_position.j2) + ','
        move_joint_cmd_str += str(joint_position.j3) + ','
        move_joint_cmd_str += str(joint_position.j4) + ','
        move_joint_cmd_str += str(joint_position.j5) + ','
        move_joint_cmd_str += str(joint_position.j6)
        return ResponseStatus.OK if self.__motion_client_s4.send_message(move_joint_cmd_str.encode()) == COMMAND_OK_BYTE else ResponseStatus.CONTROLLER_ERROR

    def check_executable_trajectory_cache(self, trajectory_dir: str, trajectory_name: str) -> ResponseStatus:
        file_manager = FileManager(save_dir=trajectory_dir)
        if file_manager.search_executable_trajectory(trajectory_name=trajectory_name):
            return ResponseStatus.OK
        else:
            return ResponseStatus.NO_FILE_FOUND
    
    def csv2executable_trajectory(self, trajectory_dir: str, trajectory_name: str) -> ResponseStatus:
        file_manager = FileManager(save_dir=trajectory_dir)
        csv_path = file_manager.search_csv_trajectory(trajectory_name=trajectory_name)
        if not csv_path:
            return ResponseStatus.NO_FILE_FOUND
        cmd_packets_bytes = FanucJ519_stream_motion_prepare(trajectory_csv=pandas.read_csv(os.path.join(csv_path)))
        with open(os.path.join(trajectory_dir, trajectory_name + '.trajectory'), 'wb') as f:
            for pb in cmd_packets_bytes:
                f.write(pb)
        if file_manager.search_executable_trajectory(trajectory_name=trajectory_name):
            return ResponseStatus.OK
        else:
            return ResponseStatus.NO_FILE_FOUND

    def execute_trajectory(self, trajectory_dir: str, trajectory_name: str) -> ResponseStatus:
        file_manager = FileManager(save_dir=trajectory_dir)
        executable_path = file_manager.search_executable_trajectory(trajectory_name=trajectory_name)
        json_trajectory_path = file_manager.search_json_trajectory(trajectory_name)
        if not (executable_path and json_trajectory_path):
            return ResponseStatus.NO_FILE_FOUND
        # move joint to start
        start = time.time()
        json_trajectory = JsonTrajectory(trajectory_name=trajectory_name, trajectory_path=json_trajectory_path)
        self.move2joint_position_async(JointPosition(joints=np.degrees(json_trajectory.first_point).tolist()))
        self.wait_until_manipulator_ready(total_wait_time=10, wait_time_step=0.05)
        self._logger.info("Fanuc: TIME for move to start of trajectory:{}".format(time.time() - start))
        # start J519 stream motion TP program
        start_j519_response = self.__motion_client_s4.send_message(START_J519.encode())
        if start_j519_response and start_j519_response == COMMAND_OK_BYTE:
            check_task_cmd_str = CHECK_TASK_DONE + "WEIYI_FLYSHOT"
            # check if J519 is running
            flyshot_done_result = self.__motion_client_s4.send_message(check_task_cmd_str.encode()).decode()
            if flyshot_done_result and flyshot_done_result[0] == COMMAND_OK and flyshot_done_result[1] == STATE_NO:
                if FanucJ519_stream_motion_execute(self.__controller_ip, executable_trajectory_path=executable_path):
                    return ResponseStatus.OK
                else:
                    self.__motion_client_s4.send_message(END_J519.encode())
        return ResponseStatus.SERVER_ERR
        

    def wait_until_program_finished(self, program_name: str, total_wait_time: float, wait_time_step: float) -> ResponseStatus:
        """check the program if 

        Args:
            program_name (str): _description_
            total_wait_time (float): _description_
            wait_time_step (float): _description_

        Returns:
            ResponseStatus: _description_
        """
        start = time.time()
        check_task_cmd_str = CHECK_TASK_DONE + program_name
        while True:
            task_done_result = self.__motion_client_s4.send_message(check_task_cmd_str.encode()).decode()
            if task_done_result and task_done_result[0] == COMMAND_OK:
                if task_done_result[1] == STATE_YES:
                    return ResponseStatus.OK
            else:
                return ResponseStatus.SERVER_ERR
            time.sleep(wait_time_step)
            if time.time() - start > total_wait_time:
                return ResponseStatus.WAITING_TIMEOUT
    
    def stop_program(self, program_name: str) -> ResponseStatus:
        stop_task_cmd_str = STOP_TASK + program_name
        if self.__motion_client_s4.send_message(stop_task_cmd_str.encode()) == COMMAND_OK_BYTE:
            return ResponseStatus.OK
        else:
            return ResponseStatus.SERVER_ERR
        
    def run_program_async(self, program_name: str) -> ResponseStatus:
        run_task_cmd_str = START_TASK + program_name
        if self.__motion_client_s4.send_message(run_task_cmd_str.encode()) == COMMAND_OK_BYTE:
            return ResponseStatus.OK
        else:
            return ResponseStatus.SERVER_ERR

if __name__ == '__main__':
    a = FanucDriver()
    print(a.csv2executable_trajectory(trajectory_dir="tests/test_cache", trajectory_name="test"))
