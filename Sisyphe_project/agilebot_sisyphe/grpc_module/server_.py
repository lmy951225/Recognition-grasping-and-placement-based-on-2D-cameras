from typing import Callable, cast, List, Union
import os
import sys

sys.path.append("..")
sys.path.append(os.getcwd())
base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)

import grpc
import agile_robot_pb2 as pb2
import agile_robot_pb2_grpc as pb2_grpc
from containers.FileManagerSingleton import FileManagerInstance
from containers.ArmSingleton import ArmInstance
from trajectories.RawTrajectory import RawTrajectory
from trajectories.TrajectoryKinematics import TrajectoryKinematics
from trajectories.MetaTrajectoryTOPPRA import MetaTrajectoryTOPPRA
from Agilebot.IR.A.status_code import StatusCodeEnum
from Agilebot.IR.A.common.const import const
from Agilebot.IR.A.sdk_classes import MotionPose, ProgramPose
from Agilebot.IR.A.file_manager import trajectoryProgram, trajectoryCsv
from Agilebot.IR.A.digital_signals import SignalType
from concurrent import futures
import time
import logging
import math
import numpy as np 
from utils.server_util import LogConnection, generate_AlgoRes_by_status, generate_fail_AlgoRes_by_log, generate_success_AlgoRes_by_log, wait_utilt_standby, move2radian_joint_pose, move2radian_joint_pose_until_standby, check_transform_status_until_success
from utils.parameter_util import TIME_STEP, POSITION_CONSTRAINT

def generate_kinematics_if_necessary(trajectory_name: str, use_cache: bool = True) -> Union[TrajectoryKinematics, bool]:
    local_raw_trajectory_path = FileManagerInstance.search_local_raw_trajectory(trajectory_name=trajectory_name)
    local_kinematics_path = FileManagerInstance.search_local_kinematics(trajectory_name=trajectory_name) if use_cache else False
    if not local_kinematics_path and local_raw_trajectory_path:
        kinematics_results = RawTrajectory(trajectory_name=trajectory_name, trajectory_path=local_raw_trajectory_path).search_kinematics_greedy()
        FileManagerInstance.upload_local_kinematics(kinematics=kinematics_results, trajectory_name=trajectory_name)
        local_kinematics_path = FileManagerInstance.search_local_kinematics(trajectory_name=trajectory_name)
    if local_kinematics_path:
        trajectory_kinematics = TrajectoryKinematics(trajectory_name=trajectory_name, trajectory_path=local_kinematics_path)
        return trajectory_kinematics
    return False

class ControllerServer(pb2_grpc.ControllerServerServicer):
    def __init__(self):
        super(ControllerServer, self).__init__()

    @LogConnection(return_type=pb2.AlgoRes, check_connection=False)
    def connect(self, request: pb2.ControllerIP, context) -> pb2.AlgoRes:
        controller_ip = request.controller_ip
        FileManagerInstance.connect(controller_ip)
        if ArmInstance.is_connected and FileManagerInstance.is_connected:
            status = ArmInstance.alarm.reset()
            if not wait_utilt_standby(wait_time=5):
                status = StatusCodeEnum.CONNECTION_TIMEOUT
        else:
            status = ArmInstance.connect(controller_ip)
            ArmInstance.speed = 1.0
        if status == StatusCodeEnum.OK:
            logging.warning("DH param results: {}".format(ArmInstance.get_DH_parameters()))
        return generate_AlgoRes_by_status(status)

    @LogConnection(return_type=pb2.AlgoRes, check_connection=True)
    def disConnect(self, request: pb2.FakeInput, context) -> pb2.AlgoRes:
        FileManagerInstance.disconnect()
        ArmInstance.disconnect()
        return generate_AlgoRes_by_status(StatusCodeEnum.OK)

    @LogConnection(return_type=pb2.AlgoRes, check_connection=True)
    def reset(self, request: pb2.FakeInput, context) -> pb2.AlgoRes:
        return generate_AlgoRes_by_status(ArmInstance.alarm.reset())

    @LogConnection(return_type=pb2.AlgoRes, check_connection=False)
    def setUpRobot(self, request, context) -> pb2.AlgoRes: # doesn't make anysense
        return generate_AlgoRes_by_status(StatusCodeEnum.OK)

    @LogConnection(return_type=pb2.GetNameReply, check_connection=True)
    def getName(self, request, context) -> pb2.GetNameReply:
        status, version_info = ArmInstance.get_arm_model_info()
        return pb2.GetNameReply(code=pb2.ErrorCode(error_code=status.code),
                                log=pb2.ResLog(res_log=status.errmsg),
                                manipulator_type=pb2.ManipulatorType(manipulator_type=version_info))
    
    @LogConnection(return_type=pb2.GetServerVersionReply, check_connection=False)
    def getServerVersion(self, request, context) -> pb2.GetServerVersionReply:
        return pb2.GetServerVersionReply(code=pb2.ErrorCode(error_code=0),
                                        log=pb2.ResLog(res_log="OK"),
                                        controller_version="0.06")

    @LogConnection(return_type=pb2.AlgoRes, check_connection=True)
    def enableRobot(self, request: pb2.FakeInput, context) -> pb2.AlgoRes:
        status = ArmInstance.execution.servo_on()
        logging.warning("real enableRobot results: {}".format(status))
        if wait_utilt_standby(wait_time=5):
            return generate_success_AlgoRes_by_log("default servo staus is on")
        else:
            return generate_AlgoRes_by_status(StatusCodeEnum.CONNECTION_TIMEOUT)

    @LogConnection(return_type=pb2.AlgoRes, check_connection=True)
    def disableRobot(self, request: pb2.FakeInput, context) -> pb2.AlgoRes:
        status = ArmInstance.execution.servo_off()
        logging.warning("real disableRobot results: {}".format(status))
        return generate_success_AlgoRes_by_log("default servo staus is on")

    @LogConnection(return_type=pb2.AlgoRes, check_connection=True)
    def setSpeedRatio(self, request: pb2.SetSpeedRatioRequest, context) -> pb2.AlgoRes:
        speed_ratio = min(max(request.speed_ratio, 0.1), 1)
        ArmInstance.speed = speed_ratio
        _, status = ArmInstance.motion.set_param(param_name="OVC", param_value=speed_ratio)
        return generate_AlgoRes_by_status(status)
    
    @LogConnection(return_type=pb2.GetIOReply, check_connection=True)
    def getIO(self, request: pb2.GetIORequest, context) -> pb2.GetIOReply:# To Do
        io_states = []
        status = StatusCodeEnum.UNSUPPORTED_SIGNAL_TYPE
        io_type = request.io_type
        io_channels_list = [i for i in request.io_channels]
        if io_type == SignalType.SIGNAL_TYPE_DO.value:
            res, status = ArmInstance.digital_signals.multi_read(signal_type=SignalType.SIGNAL_TYPE_DO,
                                                                 io_list=io_channels_list)
            if status == StatusCodeEnum.OK:
                for i in range(len(io_channels_list)):
                    io_states.append(pb2.IOState(io=io_channels_list[i], state=res[2*i + 1], io_type=SignalType.SIGNAL_TYPE_DO.value))
        return pb2.GetIOReply(code=pb2.ErrorCode(error_code=status.code),
                              log=pb2.ResLog(res_log=status.errmsg),
                              io_states=io_states)

    @LogConnection(return_type=pb2.SetIOReply, check_connection=True)
    def setIO(self, request: pb2.SetIORequest, context) -> pb2.SetIOReply:
        status: StatusCodeEnum = StatusCodeEnum.SERVER_ERR
        if len(request.io_states) == 1:
            item = request.io_states[0]
            status: StatusCodeEnum = ArmInstance.digital_signals.write(signal_type=SignalType(item.io_type),
                                                                       index=item.io,
                                                                       value=item.state)
        else:
            io_list: List[int] = list()
            for item in request.io_states:
                if SignalType(item.io_type) != SignalType.SIGNAL_TYPE_DO:
                    return pb2.SetIOReply(code=1, log="only support DO type singal for mutli write")
                io_list.append(item.io)
                io_list.append(item.state)
            status = ArmInstance.digital_signals.multi_write(signal_type=SignalType.SIGNAL_TYPE_DO, io_list=io_list)
        return pb2.SetIOReply(code=pb2.ErrorCode(error_code=status.code),
                            log=pb2.ResLog(res_log=status.errmsg))

    @LogConnection(return_type=pb2.GetJointPositionReply, check_connection=True, logging_level=logging.info)
    def getJointPosition(self, request: pb2.FakeInput, context) -> pb2.GetJointPositionReply:
        motion_pose, status = ArmInstance.motion.get_current_pose(const.JOINT)
        if not (motion_pose and (status == StatusCodeEnum.OK)):
            motion_pose = MotionPose() # define default 0 pose
            motion_pose.joint.j1 = 0
            motion_pose.joint.j2 = 0
            motion_pose.joint.j3 = 0
            motion_pose.joint.j4 = 0
            motion_pose.joint.j5 = 0
            motion_pose.joint.j6 = 0
        return pb2.GetJointPositionReply(code=pb2.ErrorCode(error_code=status.code),
                                        log=pb2.ResLog(res_log=status.errmsg),
                                        joints=pb2.JointPosition(J1=math.radians(motion_pose.joint.j1),
                                                                J2=math.radians(motion_pose.joint.j2),
                                                                J3=math.radians(motion_pose.joint.j3),
                                                                J4=math.radians(motion_pose.joint.j4),
                                                                J5=math.radians(motion_pose.joint.j5),
                                                                J6=math.radians(motion_pose.joint.j6)))

    @LogConnection(return_type=pb2.GetPoseReply, check_connection=True, logging_level=logging.info)
    def getPose(self, request: pb2.FakeInput, context) -> pb2.GetPoseReply:
        motion_pose, status = ArmInstance.motion.get_current_pose(const.CART, ucs_id=0)
        if not (motion_pose and (status == StatusCodeEnum.OK)):
            motion_pose = MotionPose() # define default 0 pose
            motion_pose.cartData.position.x = 0
            motion_pose.cartData.position.y = 0
            motion_pose.cartData.position.z = 0
            motion_pose.cartData.position.a = 0
            motion_pose.cartData.position.b = 0
            motion_pose.cartData.position.c = 0
        return pb2.GetPoseReply(code=pb2.ErrorCode(error_code=status.code),
                                log=pb2.ResLog(res_log=status.errmsg),
                                pose=pb2.CartesianPosition(x=motion_pose.cartData.position.x,
                                                            y=motion_pose.cartData.position.y,
                                                            z=motion_pose.cartData.position.z,
                                                            rx=motion_pose.cartData.position.a,
                                                            ry=motion_pose.cartData.position.b,
                                                            rz=motion_pose.cartData.position.c))

    @LogConnection(return_type=pb2.AlgoRes, check_connection=True)
    def moveJoint(self, request: pb2.JointPosition, context) -> pb2.AlgoRes:
        status = StatusCodeEnum.OK
        # check joint position limit
        joint_position = np.asarray([request.J1, request.J2, request.J3, request.J4, request.J5, request.J6])
        joint_limit_flags = np.logical_and(joint_position > POSITION_CONSTRAINT[0], joint_position < POSITION_CONSTRAINT[1])
        joints_out_of_limit = np.where(joint_limit_flags == False)[0]
        if joints_out_of_limit.size != 0:
            status = StatusCodeEnum.UNSUPPORTED_PARAMETER
            logging.warning("joints out of limit {}".format(joints_out_of_limit.tolist()))
            logging.warning("input joint position: {}".format(joint_position.tolist()))
            logging.warning("position constraints for the robot: {}".format(POSITION_CONSTRAINT.tolist()))
        else:
            # do real move
            move_status = move2radian_joint_pose_until_standby(J1=request.J1,
                                                        J2=request.J2,
                                                        J3=request.J3,
                                                        J4=request.J4,
                                                        J5=request.J5,
                                                        J6=request.J6)
            
            if move_status != StatusCodeEnum.OK:
                logging.warning("fail to move joint: {}".format(move_status.errmsg))
        return generate_AlgoRes_by_status(status)

    @LogConnection(return_type=pb2.AlgoRes, check_connection=True)
    def movePose(self, request: pb2.CartesianPosition, context) -> pb2.AlgoRes:
        motion_pose = MotionPose()
        motion_pose.pt = const.CART
        motion_pose.cartData.position.x = request.x
        motion_pose.cartData.position.y = request.y
        motion_pose.cartData.position.z = request.z
        motion_pose.cartData.position.a = request.rx
        motion_pose.cartData.position.b = request.ry
        motion_pose.cartData.position.c = request.rz
        status: StatusCodeEnum = ArmInstance.motion.move_to_pose(motion_pose, const.MOVE_JOINT)
        if status == StatusCodeEnum.OK:
            if not wait_utilt_standby(wait_time=10):
                return generate_AlgoRes_by_status(StatusCodeEnum.CONNECTION_TIMEOUT)
        return generate_AlgoRes_by_status(status)


    @LogConnection(return_type=pb2.AlgoRes, check_connection=False)
    def uploadFlyShotTraj(self, request: pb2.UploadFlyShotTrajRequest, context) -> pb2.AlgoRes:
        if request.shot_flags[0]:
            return generate_fail_AlgoRes_by_log(error_log="first point is shot flag, it's not possible for current case")
        trajectory_name = request.name
        robot_config = {"use_do": True, "io_addr": [7, 8], "io_keep_cycles": 16, "acc_limit": 1, "jerk_limit": 1}
        raw_trajectory = {"traj_waypoints": list(), "trajectory_transition_waypoints": list(), "shot_flags": list(), "offset_values": list(), "addr": list()}
        traj_waypoints = [[joints_position.J1, joints_position.J2, joints_position.J3, joints_position.J4, joints_position.J5, joints_position.J6] for joints_position in request.joints]
        # check position limit
        meta_trajectory = MetaTrajectoryTOPPRA(toppra_instance=traj_waypoints)
        position_check = meta_trajectory.check_position_limit()
        if position_check != -1:
            return generate_fail_AlgoRes_by_log(error_log="{} waypoint {} is out of the position limit {}".format(position_check, traj_waypoints[position_check], POSITION_CONSTRAINT))
        # generate raw trajectory
        for addr in request.traj_addrs:
            raw_trajectory["addr"].append([a for a in addr.addrs])
        raw_trajectory["shot_flags"] = [shot_flag for shot_flag in request.shot_flags]
        point_states = list()
        for addr_info, shot_flag in zip(raw_trajectory["addr"], raw_trajectory["shot_flags"]):
            if shot_flag:
                point_states.append("|".join([str(i) for i in addr_info]))
            else:
                point_states.append(-1)
        raw_trajectory["traj_waypoints"] = traj_waypoints
        raw_trajectory["offset_values"].append([offset for offset in request.offset])
        transition_results = meta_trajectory.generate_transition_points(point_states)
        if not transition_results:
            return generate_fail_AlgoRes_by_log(error_log="fail to add transition points to ")
        trajectory_transition_waypoints, trajectory_transition_states = transition_results
        raw_trajectory["trajectory_transition_waypoints"] = trajectory_transition_waypoints
        raw_trajectory["trajectory_transition_states"] = trajectory_transition_states
        
        whole_trajectory =  {"robot": robot_config, "flying_shots": {trajectory_name: raw_trajectory}}
        local_raw_trajectory_path = FileManagerInstance.upload_local_raw_trajectory(raw_trajectory=whole_trajectory, trajectory_name=trajectory_name)
        generate_kinematics_if_necessary(trajectory_name=trajectory_name, use_cache=False)
        return generate_success_AlgoRes_by_log(local_raw_trajectory_path)

    @LogConnection(return_type=pb2.AlgoRes, check_connection=True)
    def deleteFlyShotTraj(self, request: pb2.DeleteFlyShotTrajRequest, context) -> pb2.AlgoRes:
        FileManagerInstance.delete_local_raw_trajectory(trajectory_name=request.name)
        FileManagerInstance.delete_local_detail_trajectory(trajectory_name=request.name)
        FileManagerInstance.delete_local_kinematics(trajectory_name=request.name)
        FileManagerInstance.file_manager.delete(file_name=request.name + ".csv", file_type=trajectoryCsv)
        status:StatusCodeEnum = FileManagerInstance.file_manager.delete(file_name=request.name + "_torque.trajectory", file_type=trajectoryProgram)
        return generate_AlgoRes_by_status(status)
    
    @LogConnection(return_type=pb2.IsFlyShotTrajValidReply, check_connection=True)
    def isFlyShotTrajValid(self, request:pb2.IsFlyShotTrajValidRequest, context) -> pb2.IsFlyShotTrajValidReply:
        local_raw_trajectory_path = FileManagerInstance.search_local_raw_trajectory(trajectory_name=request.name)
        if not local_raw_trajectory_path:
            return pb2.IsFlyShotTrajValidReply(code = pb2.ErrorCode(error_code=1), 
                                            log = pb2.ResLog(res_log="no local raw trajectory find"))
        raw_trajectory = RawTrajectory(trajectory_name=request.name, trajectory_path=local_raw_trajectory_path)
        # generate kinematics if necessary  
        trajectory_kinematics = generate_kinematics_if_necessary(trajectory_name=request.name)
        if not trajectory_kinematics:
            return pb2.IsFlyShotTrajValidReply(code = pb2.ErrorCode(error_code=1), 
                                            log = pb2.ResLog(res_log="no local raw kinematics find or created, please reupload"))
        acc_rate = np.asarray(trajectory_kinematics.joint_acc_coef)
        # generate detail trajectory and check the limit
        detail_trajectory = raw_trajectory.generate_available_detail_trajectory(acc_rate=acc_rate, speed_ratio=ArmInstance.speed)  # only consider fastest 
        FileManagerInstance.upload_local_detail_trajectory(detail_trajectory, trajectory_name=request.name)
        local_detail_trajectory_path = FileManagerInstance.search_local_detail_trajectory(trajectory_name=request.name)
        # save if possible
        log = local_detail_trajectory_path
        if request.save_traj:
            upload_status: StatusCodeEnum = FileManagerInstance.file_manager.upload(file_path=local_detail_trajectory_path, file_type=trajectoryCsv, overwriting=True)
            log = log + "   upload staus:" + upload_status.errmsg
            if upload_status == StatusCodeEnum.OK:
                transfer_status, _ = ArmInstance.trajectory.transform_csv_to_trajectory(file_name=request.name + ".csv")
                log = log + " transfer status:" + transfer_status.errmsg
                if check_transform_status_until_success(request.name + "_torque.trajectory"):
                    print("success to transform from csv2trajectory")
        return pb2.IsFlyShotTrajValidReply(code = pb2.ErrorCode(error_code=0), 
                                            log = pb2.ResLog(res_log=log),
                                            traj_planning_time=str(detail_trajectory.shape[0] * TIME_STEP))
        
        
    @LogConnection(return_type=pb2.OptimizeKinematicsParamsReply, check_connection=False)
    def optimizeKinematicsParams(self, request:pb2.OptimizeKinematicsParamsRequest, context) -> pb2.OptimizeKinematicsParamsReply:
        trajectory_kinematics = generate_kinematics_if_necessary(trajectory_name=request.name)
        if trajectory_kinematics:
            kinematics_params = pb2.kinematicsParams(joint_vel_coef = trajectory_kinematics.joint_vel_coef,
                                                        joint_acc_coef = trajectory_kinematics.joint_acc_coef,
                                                        joint_jerk_coef = trajectory_kinematics.joint_jerk_coef)
            return pb2.OptimizeKinematicsParamsReply(code=pb2.ErrorCode(error_code=0), log=pb2.ResLog(res_log="OK"), kinematics_params=kinematics_params,
                                                         traj_planning_time=str(trajectory_kinematics.traj_planning_time))
        return pb2.OptimizeKinematicsParamsReply(code=pb2.ErrorCode(error_code=1), log=pb2.ResLog(res_log="fail to generate kinematics"))
    
    @LogConnection(return_type=pb2.AlgoRes, check_connection=True)
    def executeFlyShotTraj(self, request:pb2.ExecuteFlyShotTrajRequest, context) -> pb2.AlgoRes:
        name = request.name
        method = request.method # 暂时不写 
        kinematics_params = request.kinematics_params
        executable_filename = name + "_torque.trajectory"
        csv_filename = name + ".csv"

        local_raw_trajectory_path = FileManagerInstance.search_local_raw_trajectory(trajectory_name=name)
        if local_raw_trajectory_path:
            logging.warning("find local raw file {}".format(local_raw_trajectory_path))
        else:
            return generate_fail_AlgoRes_by_log(error_log="no local raw trajectory find, please update first")
        raw_trajectory_instance = RawTrajectory(trajectory_name=name, trajectory_path=local_raw_trajectory_path)
        # move to start if necessary
        if request.move_to_start:
            start_point = raw_trajectory_instance.first_point
            move_staus: StatusCodeEnum = move2radian_joint_pose(J1=start_point[0], J2=start_point[1], J3=start_point[2], J4=start_point[3], J5=start_point[4], J6=start_point[5])
            logging.warning("move to start status: {}".format(move_staus.errmsg))
        
        # prepare procedures represent local_raw -> local_csv -> remote_csv -> executable_trajectory
        trajectory_exist_flags = [True, False, False, False] 
        if request.use_cache:
            search_file_list: List[str] = list()
            # search executable trajectory file
            search_status: StatusCodeEnum = FileManagerInstance.file_manager.search(pattern=executable_filename, file_list=search_file_list)
            if (len(search_file_list) == 1) and (search_status == StatusCodeEnum.OK):
                logging.info("find corresponding executable file {}".format(executable_filename))
                trajectory_exist_flags[3] = True
            else: # no executable trajectory file
                search_file_list.clear() 
                # search remote csv file
                search_status: StatusCodeEnum = FileManagerInstance.file_manager.search(pattern=csv_filename, file_list=search_file_list)
                if (len(search_file_list) == 1) and (search_status == StatusCodeEnum.OK):
                    logging.info("find remote csv file {}".format(csv_filename))
                    trajectory_exist_flags[2] = True
                else: # no remote csv file
                    # search local csv file
                    local_detail_trajectory_path = FileManagerInstance.search_local_detail_trajectory(trajectory_name=name)
                    if local_detail_trajectory_path:
                        logging.info("find local csv file {}".format(local_detail_trajectory_path))
                        trajectory_exist_flags[1] = True
        
        # do process by trajectory file flags
        last_true_index = len(trajectory_exist_flags) - 1 - trajectory_exist_flags[::-1].index(True)
        for i in range(last_true_index, 4):
            if i == 0: # local raw -> local csv
                trajectory_kinematics = generate_kinematics_if_necessary(trajectory_name=request.name)
                acc_rate = np.asarray(trajectory_kinematics.joint_acc_coef)
                detail_trajectory = raw_trajectory_instance.generate_available_detail_trajectory(acc_rate=acc_rate, speed_ratio=ArmInstance.speed)
                FileManagerInstance.upload_local_detail_trajectory(detail_trajectory, trajectory_name=name)
            elif i == 1: # local csv -> remote csv
                local_detail_path = FileManagerInstance.search_local_detail_trajectory(trajectory_name=name)
                if local_detail_path:
                    start = time.time()
                    upload_status: StatusCodeEnum = FileManagerInstance.file_manager.upload(file_path=local_detail_path, file_type=trajectoryCsv, overwriting=True)
                    print("upload CSV time: {}".format(time.time() - start))
                    if upload_status == StatusCodeEnum.OK:
                        continue
                return generate_fail_AlgoRes_by_log(error_log="Fail to find/upload detail trajectory: {}".format(upload_status.errmsg)) 
            elif i == 2: # remote csv -> remote trajectory
                search_file_list: List[str] = list()
                search_status:StatusCodeEnum = FileManagerInstance.file_manager.search(pattern=csv_filename, file_list=search_file_list)
                if (len(search_file_list) == 1) and (search_status == StatusCodeEnum.OK):
                    ArmInstance.trajectory.transform_csv_to_trajectory(csv_filename)
                    if check_transform_status_until_success(executable_filename):
                        continue
                return generate_fail_AlgoRes_by_log(error_log="fail to find remote csv file or transfer csv to .trajectory file")
            elif i == 3:
                search_file_list: List[str] = list()
                search_status:StatusCodeEnum = FileManagerInstance.file_manager.search(pattern=executable_filename, file_list=search_file_list)
                if (len(search_file_list) == 1) and (search_status == StatusCodeEnum.OK):
                    continue
                else:
                    return generate_fail_AlgoRes_by_log(error_log="fail to find remote executable .trajectory file")
        # check move2start standby
        if not wait_utilt_standby(wait_time=20/ArmInstance.speed):
            return generate_fail_AlgoRes_by_log(error_log="fail to wait servo standby")
        # do process by trajectory file flags
        status_set:StatusCodeEnum = ArmInstance.trajectory.set_offline_trajectory_file(executable_filename)
        if status_set == StatusCodeEnum.OK:
            status_pre:StatusCodeEnum = ArmInstance.trajectory.prepare_offline_trajectory()
            if status_pre == StatusCodeEnum.OK:
                status_exe:StatusCodeEnum = ArmInstance.trajectory.execute_offline_trajectory()
                if status_exe == StatusCodeEnum.OK:
                    if wait_utilt_standby(wait_time=500 / ArmInstance.speed, wait_time_step = 0.5):
                        FileManagerInstance.file_manager.delete(file_name=csv_filename, file_type=trajectoryCsv)
                        if not request.use_cache:
                            FileManagerInstance.file_manager.delete(file_name=executable_filename, file_type=trajectoryProgram)
                    else:
                        status_exe = StatusCodeEnum.CONNECTION_TIMEOUT
                
                return generate_AlgoRes_by_status(status_exe)
            else:
                return generate_AlgoRes_by_status(status_pre)
        else:
            return generate_AlgoRes_by_status(status_set)
        
    
    @LogConnection(return_type=pb2.GetRobotProgramReply, check_connection=True)
    def getRobotProgramInfo(self, request: pb2.GetRobotProgramRequest, context) -> pb2.GetRobotProgramReply:
        joints_positions: List[pb2.JointPosition] = list()
        cartesian_positions: List[pb2.CartesianPosition] = list()
        poses, status = ArmInstance.program_register.read_all_poses(request.name)
        if poses:
            poses = cast(List[ProgramPose], poses)
            for pose in poses:
                if pose.poseData.pt == 'joint':
                    joints_pose = pb2.JointPosition(J1=pose.poseData.joint.j1, J2=pose.poseData.joint.j2, J3=pose.poseData.joint.j3,
                                                    J4=pose.poseData.joint.j4, J5=pose.poseData.joint.j5, J6=pose.poseData.joint.j6)
                    convert_pose, status = ArmInstance.program_register.convert_pose(pose, const.JOINT, const.CART)
                    if (status == StatusCodeEnum.OK) and convert_pose.poseData.pt == 'cart':
                        cartesian_pose = pb2.CartesianPosition(x=convert_pose.poseData.cartData.baseCart.position.x,
                                                                y=convert_pose.poseData.cartData.baseCart.position.y,
                                                                z=convert_pose.poseData.cartData.baseCart.position.z,
                                                                rx=convert_pose.poseData.cartData.baseCart.position.a,
                                                                ry=convert_pose.poseData.cartData.baseCart.position.b,
                                                                rz=convert_pose.poseData.cartData.baseCart.position.c)
                        joints_positions.append(joints_pose)
                        cartesian_positions.append(cartesian_pose)
                    else:
                        status = StatusCodeEnum.UNSUPPORTED_PARAMETER
                        break
                else:
                    status = StatusCodeEnum.INTERFACE_NOTIMPLEMENTED
                    break

        return pb2.GetRobotProgramReply(code=pb2.ErrorCode(error_code=status.code), log=pb2.ResLog(res_log=status.errmsg),
                                        joints=joints_positions, poses=cartesian_positions)


def run():
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    pb2_grpc.add_ControllerServerServicer_to_server(ControllerServer(), grpc_server)
    grpc_server.add_insecure_port('0.0.0.0:49999')
    print("server will start at 0.0.0.0:49999")
    # logging.info("server will start at 0.0.0.0:50002")
    grpc_server.start()
    try:
        while 1:
            time.sleep(3600)
    except KeyboardInterrupt:
        grpc_server.stop(0)


if __name__ == '__main__':
    run()

































































































































