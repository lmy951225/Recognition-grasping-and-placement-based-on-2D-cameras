from typing import Any, Union, Callable, List, cast
import logging
import math
import time
from containers.ArmSingleton import ArmInstance
from containers.FileManagerSingleton import FileManagerInstance
from Agilebot.IR.A.status_code import StatusCodeEnum
from Agilebot.IR.A.arm import ServoStatusEnum
from Agilebot.IR.A.trajectory import TransformStatusEnum
from Agilebot.IR.A.common.const import const
from Agilebot.IR.A.sdk_classes import MotionPose, ProgramPose
from grpc_module import agile_robot_pb2 as pb2

def move2radian_joint_pose(J1: float, J2: float, J3: float, J4: float, J5: float, J6: float) -> StatusCodeEnum:
    taget_pose = MotionPose()
    taget_pose.pt = const.JOINT
    taget_pose.joint.j1 = math.degrees(J1)
    taget_pose.joint.j2 = math.degrees(J2)
    taget_pose.joint.j3 = math.degrees(J3)
    taget_pose.joint.j4 = math.degrees(J4)
    taget_pose.joint.j5 = math.degrees(J5)
    taget_pose.joint.j6 = math.degrees(J6)
    return ArmInstance.motion.move_to_pose(taget_pose, const.MOVE_JOINT)

def wait_utilt_standby(wait_time: float = 0, wait_time_step = 0.5) -> bool:
    if wait_time == 0:
        return True
    count_time = 0
    while True:
        time.sleep(wait_time_step)
        ret, state = ArmInstance.get_servo_status()
        if ret == StatusCodeEnum.OK and state == ServoStatusEnum.SERVO_IDLE:
            return True
        if count_time > wait_time:
            return False
        count_time += wait_time_step

def move2radian_joint_pose_until_standby(J1: float, J2: float, J3: float, J4: float, J5: float, J6: float) -> StatusCodeEnum:
    move_status = move2radian_joint_pose(J1, J2, J3, J4, J5, J6)
    if move_status == StatusCodeEnum.OK:
        time.sleep(0.5)
        if not wait_utilt_standby(wait_time=10 / ArmInstance.speed):
            return StatusCodeEnum.CONNECTION_TIMEOUT 
    return move_status

def check_transform_status_until_success(trajectory_file_name: str, wait_time: float = 1000, wait_time_step = 1):
    count_time = 0
    result = False
    start = time.time()
    while True:
        status, ret = ArmInstance.trajectory.check_transform_status(trajectory_file_name)
        status = cast(StatusCodeEnum, status)
        ret = cast(TransformStatusEnum, ret)
        print("transfer csv2trajectory status: {}".format(status.errmsg))
        if status == TransformStatusEnum.TRANSFORM_SUCCESS:
            result = True
            break
        elif status in [TransformStatusEnum.TRANSFORM_FAILED, TransformStatusEnum.TRANSFORM_NOT_FOUND, TransformStatusEnum.TRANSFORM_NOT_FOUND]:
            break
        count_time += wait_time_step
        time.sleep(wait_time_step)
        if count_time > wait_time:
            break
    print("total time for transformation:{}".format(time.time() - start))
    return result

class LogConnection():
    def __init__(self, return_type: Union[pb2.AlgoRes, pb2.GetNameReply, pb2.GetIOReply, pb2.SetIOReply, pb2.IsFlyShotTrajValidReply, pb2.GetRobotProgramReply, pb2.OptimizeKinematicsParamsReply], check_connection: bool = True, logging_level: Callable = logging.warning) -> None:
        self.__return_type = return_type
        self.__check_connection = check_connection
        self.__logging_level = logging_level

    def __call__(self, func: Callable) -> Any:
        def new_func(*args, **kwargs):
            self.__logging_level(func.__name__ + " FUNCTION start to process:")
            self.__logging_level("all parameters: {}".format(args))
            start = time.time()
            # check connection if necessary
            if self.__check_connection and not (ArmInstance.is_connected and FileManagerInstance.is_connected):
                result = self.__return_type(code=pb2.ErrorCode(error_code=1), log=pb2.ResLog(res_log="connection error"))
            else:
                result = func(*args, **kwargs)
            end = time.time()
            self.__logging_level("RESULTS: {}".format(result))
            self.__logging_level(func.__name__ + " FUNCTION finish time: " + str(end-start))
            return result
        return new_func
    
def generate_AlgoRes_by_status(status: StatusCodeEnum) -> pb2.AlgoRes:
    return pb2.AlgoRes(code=pb2.ErrorCode(error_code=status.code),
                        log=pb2.ResLog(res_log=str(status.errmsg)))

def generate_success_AlgoRes_by_log(log: str) -> pb2.AlgoRes:
    return pb2.AlgoRes(code=pb2.ErrorCode(error_code=0),
                        log=pb2.ResLog(res_log=log))

def generate_fail_AlgoRes_by_log(error_log: str, error_code: int = 1):
    return pb2.AlgoRes(code=pb2.ErrorCode(error_code=error_code),
                        log=pb2.ResLog(res_log=error_log))