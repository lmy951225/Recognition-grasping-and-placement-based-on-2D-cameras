from typing import List, Union, Callable, cast, Tuple, Dict
import os
import time
import json
import zipfile
from xml.etree import ElementTree as ET
import numpy as np
from aenum import extend_enum
from Agilebot.IR.A.arm import Arm, ServoStatusEnum, RobotStatusEnum, CtrlStatusEnum
from Agilebot.IR.A.status_code import StatusCodeEnum
from Agilebot.IR.A.motion import ParamType
from Agilebot.IR.A.digital_signals import SignalType
from Agilebot.IR.A.common.const import const
from Agilebot.IR.A.sdk_classes import MotionPose, ProgramPose
from Agilebot.IR.A.file_manager import FileManager as AgiletbotFileManger
from Agilebot.IR.A.file_manager import TRAJECTORY, USER_PROGRAM
from Agilebot.IR.A.flyshot import FlyShot, DH
from RobotDriverCore.ManipulatorDriver.ManipulatorDriver import ManipulatorDriver
from RobotDriverCore.RobotDriverCoreUtils.ResponseStatus import ResponseStatus
from RobotDriverCore.RobotDriverCoreUtils.RobotConstraints import RobotConstraints
from RobotDriverCore.RobotDriverCoreUtils.RobotPosition import CartesianPosition, JointPosition
from RobotDriverCore.RobotDriverCoreUtils.SignalIO import SignalIOType
from RobotDriverCore.TrajectoryPlanner.FileManager import FileManager


def jointPosition2AgilebotPorgramPose(joint_position: JointPosition) -> ProgramPose:
    jp = ProgramPose()
    jp.id = 1
    jp.poseData.pt = "joint"
    jp.poseData.cartData.uf = 0
    jp.poseData.cartData.tf = 0
    jp.poseData.joint.j1 = joint_position.j1
    jp.poseData.joint.j2 = joint_position.j2
    jp.poseData.joint.j3 = joint_position.j3
    jp.poseData.joint.j4 = joint_position.j4
    jp.poseData.joint.j5 = joint_position.j5
    jp.poseData.joint.j6 = joint_position.j6
    jp.poseData.joint.j7 = 0
    jp.poseData.joint.j8 = 0
    jp.poseData.joint.j9 = 0
    return jp

def AgilebotStatus2RobotDriverResponse(func: Callable) -> Callable:
    """update status_code of Agilebot SDK to ResponseStatus of RobotDriverCore

    Args:
        func (Callable): function to enable descriptor

    Returns:
        Callable: return the final result
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, StatusCodeEnum):
            if result.value not in ResponseStatus._value2member_map_:
                extend_enum(ResponseStatus, result.errmsg, (result.code, result.errmsg))                
            return ResponseStatus(result.value)
        else:
            return result
    
    return wrapper

class AgilebotDriver(ManipulatorDriver):
    def __init__(self) -> None:
        super().__init__()
        self.__arm = Arm()
        self.__AgilebotFileManager = None
        self._flyshot = FlyShot()

    def __sync_flyshot_DH(self) -> StatusCodeEnum:
        """sync the DH parameters from Agilebot Manipulator to RobotDriverCore environment

        Returns:
            StatusCodeEnum: sync status
        """
        dh_list, ret = self.__arm.motion.get_DH_param()
        if ret != StatusCodeEnum.OK:
            return ret
        base_dh = DH()
        base_dh.d = dh_list[0].d
        base_dh.a = dh_list[0].a
        base_dh.alpha = dh_list[0].alpha
        base_dh.offset = dh_list[0].offset
        arm_dh_arr = [
            DH(dh_list[1].d, dh_list[1].a, dh_list[1].alpha, dh_list[1].offset),
            DH(dh_list[2].d, dh_list[2].a, dh_list[2].alpha, dh_list[2].offset),
            DH(dh_list[3].d, dh_list[3].a, dh_list[3].alpha, dh_list[3].offset),
            DH(dh_list[4].d, dh_list[4].a, dh_list[4].alpha, dh_list[4].offset),
            DH(dh_list[5].d, dh_list[5].a, dh_list[5].alpha, dh_list[5].offset),
            DH(dh_list[6].d, dh_list[6].a, dh_list[6].alpha, dh_list[6].offset)
        ]
        return self._flyshot.set_DH(base_dh, arm_dh_arr)

    def _generate_robot_constraints(self) -> RobotConstraints:
        """Agilebot constraints is from Agilebot company 

        Returns:
            RobotConstraints: constraints of Agilebt P7A-900
        """
        robot_constraint = RobotConstraints() # unit is radian
        robot_constraint.time_step = 0.001
        # different limitS for                           J1     J2      J3      J4      J5      J6
        robot_constraint.max_position       =np.asarray([ 2.94, 1.72, 3.47,   3.29,  1.98,  6.26])
        robot_constraint.min_position       =np.asarray([-2.94,-2.33, -1.2,  -3.29, -1.98, -6.26])
        robot_constraint.velocity_limit     =np.asarray([5.71,  4.56, 5.71,   7.75,  6.97, 10.46])
        robot_constraint.acceleration_limit =np.asarray([23,      19,   23,     31,    27,    39])
        robot_constraint.jerk_limit         =np.asarray([23,      19,   23,     31,    27,    39]) * 8
        return robot_constraint
    
    def connect(self, controller_ip: str) -> StatusCodeEnum:
        """connect the Agilebot Manipulator and sync DH parameters
            for flyshot function, please sync the flyshot configs in child classes
        Args:
            controller_ip (str): robot ip to connect

        Returns:
            StatusCodeEnum: connect results
        """
        # do connect without sync flyshot config
        # sync flyshot config in child classes
        self.__AgilebotFileManager = AgiletbotFileManger(controller_ip)
        connect_status = self.__arm.connect(controller_ip)
        if connect_status != StatusCodeEnum.OK:
            return connect_status
        return self.__sync_flyshot_DH()
    
    def disconnect(self) -> ResponseStatus:
        self.__arm.disconnect()
        return ResponseStatus.OK
    
    def is_connect(self) -> ResponseStatus:
        if self.__arm.is_connect():
            return ResponseStatus.OK
        else:
            return ResponseStatus.CONNECTION_TIMEOUT
        
    @AgilebotStatus2RobotDriverResponse
    def forward_kinematics(self, joint_position: JointPosition) -> Union[ResponseStatus, CartesianPosition]:
        """transfer joint position to cartesian position, make up ProgramPose by joint position and then transfer to cartesian position

        Args:
            joint_position (JointPosition): input joint position to for Agilebot manipulator

        Returns:
            Union[ResponseStatus, CartesianPosition]: converted cartesian position or failure response
        """
        jp = jointPosition2AgilebotPorgramPose(joint_position)
        convert_pose, convert_status = self.__arm.program_register.convert_pose(jp, const.JOINT, const.CART)
        if convert_status != StatusCodeEnum.OK:
            return convert_pose
        x = convert_pose.poseData.cartData.baseCart.position.x
        y = convert_pose.poseData.cartData.baseCart.position.y
        z = convert_pose.poseData.cartData.baseCart.position.z
        rx = convert_pose.poseData.cartData.baseCart.position.a
        ry = convert_pose.poseData.cartData.baseCart.position.b
        rz = convert_pose.poseData.cartData.baseCart.position.c
        return CartesianPosition(xyzrxryrz=[x, y, z, rx, ry, rz])

    @AgilebotStatus2RobotDriverResponse
    def inverse_kinematics(self, cartesian_position: CartesianPosition, reference_joint_position: JointPosition) -> Union[ResponseStatus, JointPosition]:
        """transfer cartesian position to joint position, reference joint position is used to get the 

        Args:
            cartesian_position (CartesianPosition): input cartesian position for Agilebot manipulator
            reference_joint_position (JointPosition): reference joint position to choose the best inverrse kinematiccs results

        Returns:
            Union[ResponseStatus, JointPosition]: converted joint position or failure response
        """
        reference_jp = jointPosition2AgilebotPorgramPose(reference_joint_position)
        reference_cp, convert_status = self.__arm.program_register.convert_pose(reference_jp, const.JOINT, const.CART)
        if convert_status != StatusCodeEnum.OK:
            return convert_status
        reference_cp.poseData.cartData.baseCart.position.x = cartesian_position.x
        reference_cp.poseData.cartData.baseCart.position.y = cartesian_position.y
        reference_cp.poseData.cartData.baseCart.position.z = cartesian_position.z
        reference_cp.poseData.cartData.baseCart.position.a = cartesian_position.rx
        reference_cp.poseData.cartData.baseCart.position.b = cartesian_position.ry
        reference_cp.poseData.cartData.baseCart.position.c = cartesian_position.rz
        target_jp, convert_status = self.__arm.program_register.convert_pose(reference_cp, const.CART, const.JOINT)
        if convert_status != StatusCodeEnum.OK:
            return convert_status
        j1 = target_jp.poseData.joint.j1 
        j2 = target_jp.poseData.joint.j2 
        j3 = target_jp.poseData.joint.j3 
        j4 = target_jp.poseData.joint.j4 
        j5 = target_jp.poseData.joint.j5 
        j6 = target_jp.poseData.joint.j6
        return JointPosition(joints=[j1, j2, j3, j4, j5, j6]) 

    @AgilebotStatus2RobotDriverResponse       
    def reset(self) -> ResponseStatus:
        return self.__arm.alarm.reset()
    
    @AgilebotStatus2RobotDriverResponse
    def get_name(self) -> Union[str, ResponseStatus]:
        status, version_info = self.__arm.get_arm_model_info()
        if status == StatusCodeEnum.OK:
            return version_info
        else:
            return status
        
    def enable(self) -> ResponseStatus:
        status = self.__arm.execution.servo_on()
        self._logger.info("real Agilebot enable results: {}".format(status))
        return ResponseStatus.OK
    
    def disable(self) -> ResponseStatus:
        status = self.__arm.execution.servo_off()
        self._logger.info("real Agilebot disable results: {}".format(status))
        return ResponseStatus.OK
    
    @AgilebotStatus2RobotDriverResponse
    def set_speed_ratio(self, speed_ratio: float) -> ResponseStatus:
        speed_ratio = min(max(speed_ratio, 0.1), 1)
        self._speed_ratio = speed_ratio
        return self.__arm.motion.set_param(param_name=ParamType.OVC, param_value=speed_ratio)
        
    @AgilebotStatus2RobotDriverResponse
    def get_io(self, signal_type: SignalIOType, ports: List[int]) -> Union[List[bool], ResponseStatus]:
        type_code = signal_type.Agilebot_code
        if type_code <= 0 or type_code > 8:
            return ResponseStatus.UNSUPPORTED_SIGNAL_TYPE
        if signal_type == SignalIOType.SIGNAL_DO:
            res, status = self.__arm.digital_signals.multi_read(signal_type=SignalType.SIGNAL_TYPE_DO, io_list=ports)
            if status == StatusCodeEnum.OK:
                return [True if res[2*i + 1] == 1 else False for i in range(len(ports))]
        else:
            results: List[bool] = list()
            for one_port in ports:
                res, status = self.__arm.digital_signals.read(signal_type=SignalType(signal_type.Fanuc_code), index=one_port)
                if status == StatusCodeEnum.OK:
                    results.append(True if res == 1 else False)
                else:
                    break
            return results
        return status
    
    @AgilebotStatus2RobotDriverResponse
    def set_io(self, signal_type: SignalIOType, ports: List[int], io_value: List[bool]) -> ResponseStatus:
        status = ResponseStatus.SERVER_ERR
        if len(ports) == 1:
            status = self.__arm.digital_signals.write(signal_type=SignalType(signal_type.Agilebot_code), index=ports[0], value=io_value[0])
        else:
            io_list: List[int] = list()
            if signal_type.Fanuc_code != SignalType.SIGNAL_TYPE_DO.value:
                return ResponseStatus.UNSUPPORTED_SIGNAL_TYPE
            for one_port, one_value in zip(ports, io_value):
                io_list.append(one_port)
                io_list.append(1 if one_value else 0)
            status = self.__arm.digital_signals.multi_write(signal_type=SignalType.SIGNAL_TYPE_DO, io_list=io_list)
        return status
    
    @AgilebotStatus2RobotDriverResponse
    def get_DH_parameters(self) -> Union[List, ResponseStatus]:
        ret, dh_list = self._flyshot.get_DH()
        dh_parameters: List[List[float]] = list()
        if ret == StatusCodeEnum.OK:
            for dh in dh_list:
                dh = cast(DH, dh)
                dh_parameters.append([dh.a / 1000, dh.alpha, dh.d / 1000, dh.offset])
            return dh_parameters
        else:
            return ret
    
    @AgilebotStatus2RobotDriverResponse
    def get_current_joint_position(self) -> Union[JointPosition, ResponseStatus]:
        motion_pose, status = self.__arm.motion.get_current_pose(const.JOINT)
        if status == StatusCodeEnum.OK:
            joint_position = JointPosition(joints=[motion_pose.joint.j1,
                                                   motion_pose.joint.j2,
                                                   motion_pose.joint.j3,
                                                   motion_pose.joint.j4,
                                                   motion_pose.joint.j5,
                                                   motion_pose.joint.j6,])
            return joint_position
        else:
            return status
        
    @AgilebotStatus2RobotDriverResponse    
    def get_current_cartesian_position(self) -> Union[CartesianPosition, ResponseStatus]:
        motion_pose, status = self.__arm.motion.get_current_pose(const.CART, ucs_id = 0)
        if status == StatusCodeEnum.OK:
            cartesian_position = CartesianPosition(xyzrxryrz=[motion_pose.cartData.position.x,
                                                              motion_pose.cartData.position.y,
                                                              motion_pose.cartData.position.z,
                                                              motion_pose.cartData.position.a,
                                                              motion_pose.cartData.position.b,
                                                              motion_pose.cartData.position.c])
            return cartesian_position
        else:
            return status
        
    @AgilebotStatus2RobotDriverResponse
    def wait_until_manipulator_ready(self, total_wait_time: float, wait_time_step: float) -> ResponseStatus:
        start = time.time()
        while True:
            # check ctrl status
            ctrl_ret, ctrl_state = self.__arm.get_ctrl_status()
            if ctrl_ret != StatusCodeEnum.OK:
                return ctrl_ret
            # check robot status
            robot_ret, robot_state = self.__arm.get_robot_status()
            if robot_ret != StatusCodeEnum.OK:
                return robot_ret
            # check servo status
            servo_ret, servo_state = self.__arm.get_servo_status()
            if servo_ret != StatusCodeEnum.OK:
                return servo_ret
            # check ctrl state
            if ctrl_state == CtrlStatusEnum.CTRL_ESTOP:
                return StatusCodeEnum.SERVER_ERR
            # check idle
            if servo_state == ServoStatusEnum.SERVO_IDLE and robot_state == RobotStatusEnum.ROBOT_IDLE:
                return StatusCodeEnum.OK
            # check time out
            time.sleep(wait_time_step)
            if time.time() - start > total_wait_time:
                return ResponseStatus.WAITING_TIMEOUT
            

    @AgilebotStatus2RobotDriverResponse
    def move2joint_position_async(self, joint_position: JointPosition) -> ResponseStatus:
        target_pose = MotionPose()
        target_pose.pt = const.JOINT
        target_pose.joint.j1 = joint_position.j1
        target_pose.joint.j2 = joint_position.j2
        target_pose.joint.j3 = joint_position.j3
        target_pose.joint.j4 = joint_position.j4
        target_pose.joint.j5 = joint_position.j5
        target_pose.joint.j6 = joint_position.j6
        return self.__arm.motion.move_to_pose(target_pose, const.MOVE_JOINT)
    
    def check_executable_trajectory_cache(self, trajectory_dir: str, trajectory_name: str) -> ResponseStatus:
        search_file_list: List[str] = list()
        search_status = self.__AgilebotFileManager.search(pattern=trajectory_name+'.trajectory', file_list=search_file_list)
        if (len(search_file_list) == 1) and (search_status == StatusCodeEnum.OK):
            return ResponseStatus.OK
        else:
            return ResponseStatus.NO_FILE_FOUND
        
    @AgilebotStatus2RobotDriverResponse
    def csv2executable_trajectory(self, trajectory_dir: str, trajectory_name: str) -> ResponseStatus:
        """transform csv to excutable trajectory than upload to Agilebot Robot

        Args:
            trajectory_dir (str): directory to save trajectory
            trajectory_name (str): name of trajectory, also file prefix for correponding trajectory

        Returns:
            ResponseStatus: process results
        """
        file_manager = FileManager(save_dir=trajectory_dir)
        csv_path = file_manager.search_csv_trajectory(trajectory_name=trajectory_name)
        if not csv_path:
            return ResponseStatus.NO_FILE_FOUND
        executable_path = os.path.join(trajectory_dir, trajectory_name + ".trajectory") 
        csv2trajectory_status = self._flyshot.csv2trajectory(csv_file=csv_path.encode('utf-8'), trajectory_file=executable_path.encode('utf-8'))
        if not csv2trajectory_status == StatusCodeEnum.OK:
            return ResponseStatus.CSV2TRAJECTORY_FAIL
        return self.__AgilebotFileManager.upload(file_path=os.path.join(trajectory_dir, executable_path), file_type=TRAJECTORY, overwriting=True)

    @AgilebotStatus2RobotDriverResponse
    def execute_trajectory(self, trajectory_dir: str, trajectory_name: str) -> ResponseStatus:
        executable_filename = trajectory_name + ".trajectory"
        # set offline trajectory
        status_set = self.__arm.trajectory.set_offline_trajectory_file(executable_filename)
        if status_set != StatusCodeEnum.OK:
            return status_set
        # prepare offline trajectory
        status_pre = self.__arm.trajectory.prepare_offline_trajectory()
        if status_pre != StatusCodeEnum.OK:
            return status_pre
        start = time.time()
        prepare_wait = self.wait_until_manipulator_ready(total_wait_time = 10/self._speed_ratio, wait_time_step = 0.05) # for the case that smaller than start-threshold but still need to move to start
        if prepare_wait != ResponseStatus.OK:
            return prepare_wait
        self._logger.info("Agilebot: TIME for prepare offline trajectory:{}".format(time.time() - start))
        # execute trajectory
        return self.__arm.trajectory.execute_offline_trajectory()
    
    @AgilebotStatus2RobotDriverResponse
    def stop_program(self, program_name: str) -> ResponseStatus:
        return self.__arm.execution.stop(program_name)
    
    @AgilebotStatus2RobotDriverResponse
    def wait_until_program_finished(self, program_name: str, total_wait_time: float, wait_time_step: float) -> ResponseStatus:
        start = time.time()
        while True:
            running_status, program_list = self.__arm.execution.all_running_programs()
            if running_status != StatusCodeEnum.OK:
                return running_status
            if program_name not in program_list.values():
                return StatusCodeEnum.OK
            time.sleep(wait_time_step)
            if time.time() - start > total_wait_time:
                return ResponseStatus.WAITING_TIMEOUT
    
    @AgilebotStatus2RobotDriverResponse
    def run_program_async(self, program_name: str) -> ResponseStatus:
        self.__arm.execution.stop(program_name)
        return self.__arm.execution.start(program_name)

    @AgilebotStatus2RobotDriverResponse
    def read_program_info(self, program_name: str) -> Union[ResponseStatus, Tuple[List[JointPosition], List[CartesianPosition]]]:
        download_status = self.__AgilebotFileManager.download(file_name=program_name, file_path=program_name, file_type=USER_PROGRAM, overwriting=True)
        if download_status != StatusCodeEnum.OK:
            return download_status
        with zipfile.ZipFile(program_name + '.zip', 'r') as zf:
            zf.extractall()
        #################################read program json###################################
        joint_positions: Dict[int, List[JointPosition]] = dict()
        cartesian_positions: Dict[int, List[CartesianPosition]] = dict()
        with open(program_name + '.json', 'r') as f:
            program_json = json.load(f)
        poses = program_json.get('poses', list())
        for pose in poses:
            p = ProgramPose()
            p.generate_program_pose_from_dict(pose)
            if p.poseData.pt == 'joint':
                joint_positions[pose['id']] = JointPosition(joints=[p.poseData.joint.j1,p.poseData.joint.j2,p.poseData.joint.j3,p.poseData.joint.j4,p.poseData.joint.j5,p.poseData.joint.j6])
                convert_pose, convert_status = self.__arm.program_register.convert_pose(p, const.JOINT, const.CART)
                if (convert_status == StatusCodeEnum.OK) and convert_pose:
                    cartesian_positions[pose['id']] = CartesianPosition(xyzrxryrz=[convert_pose.poseData.cartData.baseCart.position.x,convert_pose.poseData.cartData.baseCart.position.y,convert_pose.poseData.cartData.baseCart.position.z,
                                                                            convert_pose.poseData.cartData.baseCart.position.a,convert_pose.poseData.cartData.baseCart.position.b,convert_pose.poseData.cartData.baseCart.position.c])
                else:
                    return StatusCodeEnum.PROGRAM_POSE_NOT_FOUND
            elif p.poseData.pt == 'cart':
                cartesian_positions[pose['id']] = CartesianPosition(xyzrxryrz=[p.poseData.cartData.baseCart.position.x,p.poseData.cartData.baseCart.position.y,p.poseData.cartData.baseCart.position.z,
                                                                        p.poseData.cartData.baseCart.position.a,p.poseData.cartData.baseCart.position.b,p.poseData.cartData.baseCart.position.z])
                convert_pose, convert_status = self.__arm.program_register.convert_pose(p, const.CART, const.JOINT)
                if (convert_status == StatusCodeEnum.OK) and convert_pose:
                    joint_positions[pose['id']] = JointPosition(joints=[convert_pose.poseData.joint.j1, convert_pose.poseData.joint.j2, convert_pose.poseData.joint.j3,
                                                                    convert_pose.poseData.joint.j4, convert_pose.poseData.joint.j5, convert_pose.poseData.joint.j6])
                else:
                    return StatusCodeEnum.PROGRAM_POSE_NOT_FOUND
        #################################read program json###################################
        tag_indices = list()
        def dfs(e):
            if e.tag == 'argument' and e.attrib.get('name', "") == 'pose' and e.attrib.get('type', "") == 'pose':
                for c in e:
                    if c.tag == 'element' and c.attrib.get('type', "") == 'num':
                        tag_indices.append(int(c.text))
            for child in e:
                dfs(child)
        
        program_xml = ET.parse(program_name + '.xml')
        root = program_xml.getroot()
        dfs(root)
        ################################merge results from json and xml############################
        jps:List[JointPosition]= list()
        cps:List[CartesianPosition]= list()
        for tag in tag_indices:
            if tag in joint_positions and tag in cartesian_positions:
                jps.append(joint_positions[tag])
                cps.append(cartesian_positions[tag])
            else:
                return StatusCodeEnum.PROGRAM_POSE_NOT_FOUND
        return jps, cps
        