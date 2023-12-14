from typing import List, Union, Tuple
import logging
import time
from RobotDriverCore.RobotDriverCoreUtils.ResponseStatus import ResponseStatus
from RobotDriverCore.RobotDriverCoreUtils.SignalIO import SignalIOType
from RobotDriverCore.RobotDriverCoreUtils.RobotPosition import JointPosition, CartesianPosition
from RobotDriverCore.RobotDriverCoreUtils.RobotConstraints import RobotConstraints


class ManipulatorDriver():
    def __init__(self) -> None:
        self._speed_ratio = 0.5 # default speed
        self._soft_enable = False # # default able condition
        self._robot_constraint: RobotConstraints = self._generate_robot_constraints()
        self._logger = logging.getLogger('ManipulatorRobotDriver')
    
    def _generate_robot_constraints(self) -> RobotConstraints:
        """generate robot constraints, different robot with different constraint
            Only initialize once and will never updated
        Returns:
            RobotConstraints: instance for RobotConstraints
        """
        raise NotImplementedError
    
    @property
    def robot_constraints(self) -> RobotConstraints:
        return self._robot_constraint

    def connect(self, controller_ip: str) -> ResponseStatus: 
        """connect to the manipulator

        Args:
            controller_ip (str): ip address of the manipulator

        Returns:
            ResponseStatus: connect result
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED 

    def disconnect(self) -> ResponseStatus:
        """disconnect to the manipulator

        Returns:
            ResponseStatus: disconnect result
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED 
    
    def forward_kinematics(self, joint_position: JointPosition) -> Union[ResponseStatus, CartesianPosition]:
        """transfer joint position to cartesian position

        Args:
            joint_position (JointPosition): input joint parameters

        Returns:
            Union[ResponseStatus, CartesianPosition]: correponding cartesian position or failure response
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED

    def inverse_kinematics(self, cartesian_position: CartesianPosition, reference_joint_position: JointPosition) -> Union[ResponseStatus, JointPosition]:
        """transfer cartesian position to joint position with reference joint position, 
            since inverse kinematics may have multiple results

        Args:
            cartesian_position (CartesianPosition): input cartesian parameters
            reference_joint_position (JointPosition): reference paramtere

        Returns:
            Union[ResponseStatus, JointPosition]: correponding joint position or failure response
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED

    def is_connect(self) -> ResponseStatus:
        """check the connect status

        Returns:
            ResponseStatus: connection check result
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED 
    
    def reset(self) -> ResponseStatus:
        """reset the robot 

        Returns:
            ResponseStatus: reset result
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED
    
    def get_name(self) -> Union[str, ResponseStatus]:
        """get name of the manipulator

        Returns:
            Union[str, ResponseStatus]: manipulator name or failure response
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED 

    def enable(self) -> ResponseStatus:
        """power/servo on the manipulator

        Returns:
            ResponseStatus: power/servo on result
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED 

    def disable(self) -> ResponseStatus:
        """power/servo off the manipulator 

        Returns:
            ResponseStatus: power/servo off results
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED 

    def set_speed_ratio(self, speed_ratio: float) -> ResponseStatus:
        """set speed ratio between 0.1(slow) and 1(fast)


        Returns:
            ResponseStatus: set result
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED 

    @property
    def speed_ratio(self) -> float:
        """get speed ratio

        Returns:
            float: speed ratio
        """
        return self._speed_ratio
    
    def get_io(self, signal_type:SignalIOType, ports: List[int]) -> Union[List[bool], ResponseStatus]:
        """get io information 

        Args:
            signal_type (SignalIOType): io type
            ports (List[int]): ports number for io

        Returns:
            Union[List[bool], ResponseStatus]: io information or failure response
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED 

    def set_io(self, signal_type:SignalIOType, ports: List[int], io_value: List[bool]) -> ResponseStatus:
        """set io with ports and value

        Args:
            signal_type (SignalIOType): io type
            ports (List[int]):  ports number for io
            io_value (List[bool]): io value as ports list

        Returns:
            ResponseStatus: set result
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED 
    
    def get_DH_parameters(self) -> Union[List, ResponseStatus]:
        """get specific dh parameters for the connected robots

        Returns:
            Union[List, ResponseStatus]: List for each joint with [a, alpha, d, offset] or failure response
] 
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED

    def get_current_joint_position(self) -> Union[JointPosition, ResponseStatus]:
        """get current information of joint position

        Returns:
            Union[JointPosition, ResponseStatus]: joint position format(j1,j2.....j7,j8,j9) or failure response
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED 

    def move2joint_position_async(self, joint_position: JointPosition) -> ResponseStatus:
        """move manipulator to joint position without waiting stand by

        Args:
            joint_position (JointPosition): move target of the joint position

        Returns:
            ResponseStatus: move result
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED 
    
    def move2joint_position_sync(self, joint_position: JointPosition) -> ResponseStatus:
        """move manipulator to joint position until stand by

        Args:
            joint_position (JointPosition): move target of the joint position

        Returns:
            ResponseStatus: move result
        """
        move_result = self.move2joint_position_async(joint_position=joint_position)
        if move_result == ResponseStatus.OK:
            time.sleep(0.1)
            return self.wait_until_manipulator_ready(total_wait_time=10/self._speed_ratio, wait_time_step=0.1)
        else:
            return move_result

    def get_current_cartesian_position(self) -> Union[CartesianPosition, ResponseStatus]:
        """get current information of cartesian position

        Returns:
            Union[CartesianPosition, ResponseStatus]: cartesian position format(x,y,z,rx,ry,rz) or failure result
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED 
    
    def move_line_async(self, position: Union[JointPosition, CartesianPosition]) -> ResponseStatus:
        """move linear motion of tcp without waiting standby

        Args:
            position (Union[JointPosition, CartesianPosition]): move result, maybe joint position or cartesian position

        Returns:
            ResponseStatus: move result
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED 
    
    def move_line_sync(self, position: Union[JointPosition, CartesianPosition]) -> ResponseStatus:
        """move linear motion of tcp until stand by

        Args:
            position (Union[JointPosition, CartesianPosition]): _description_

        Returns:
            ResponseStatus: _description_
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED 
    
    def wait_until_manipulator_ready(self, total_wait_time: float, wait_time_step: float) -> ResponseStatus:
        """check manipulator states until ready to move

        Args:
            total_wait_time (float): total_time to wait the manipulator ready
            wait_time_step (float): check state interval

        Returns:
            ResponseStatus: wait result, OK or bad condition of the manipulator
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED
    
    def check_executable_trajectory_cache(self, trajectory_dir: str, trajectory_name: str) -> ResponseStatus:
        """check if the executable trajectory exist

        Args:
            trajectory_dir (str):  path dir for saving all trajectory files
            trajectory_name (str): trajectory name

        Returns:
            ResponseStatus: check cache result
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED
    
    def csv2executable_trajectory(self, trajectory_dir: str, trajectory_name: str) -> ResponseStatus:
        """check executable trajectory cache

        Args:
            trajectory_dir (str): path dir for saving all trajectory files
            traject_name (str): trajectory name

        Returns:
            ResponseStatus: transfrom result
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED
    
    def execute_trajectory(self, trajectory_dir: str, trajectory_name: str) -> ResponseStatus:
        """execute trajectory, make sure the robot position is in the start point of the trajectory

        Args:
            trajectory_dir (str): path dir for saving all trajectory files
            trajectory_name (str): trajectory name
            use_cache (bool): use chche flag

        Returns:
            ResponseStatus: execute results
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED
    
    def read_program_info(self, program_name: str) -> Union[ResponseStatus, Tuple[List[JointPosition], List[CartesianPosition]]]:
        """read positions information in robot program

        Args:
            program_name (str): name of the program

        Returns:
            Union[ResponseStatus, List[JointPosition]]: joint positions if OK or failure results
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED
    
    def run_program_async(self, program_name: str) -> ResponseStatus:
        """start to run program, return if the command sendding process is good

        Args:
            program_name (str): name of the program

        Returns:
            ResponseStatus: start command result
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED

    def run_program_sync(self, program_name: str) -> ResponseStatus:
        """running program, until finished

        Args:
            program_name (str): name of the program

        Returns:
            ResponseStatus: running program results
        """
        start_result = self.run_program_async(program_name)
        if start_result == ResponseStatus.OK:
            return self.wait_until_program_finished(program_name, total_wait_time=60, wait_time_step=1)
        else:
            return start_result 
        
    def stop_program(self, program_name: str) -> ResponseStatus:
        """stop/abort/exit program

        Args:
            program_name (str): name of the program

        Returns:
            ResponseStatus: result of stop/abort/exit
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED


    def wait_until_program_finished(self, program_name: str, total_wait_time: float, wait_time_step: float) -> ResponseStatus:
        """check the program is finied

        Args:
            program_name (str): name of the program

        Returns:
            ResponseStatus: program running finish result
        """
        return ResponseStatus.INTERFACE_NOTIMPLEMENTED