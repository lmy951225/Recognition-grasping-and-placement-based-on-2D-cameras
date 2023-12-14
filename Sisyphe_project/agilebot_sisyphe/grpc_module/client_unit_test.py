from typing import List, Dict
import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
import time
from concurrent import futures
import math
import logging
import numpy as np
import grpc
import agile_robot_pb2 as pb2
import agile_robot_pb2_grpc as pb2_grpc
from utils.path_util import docker_host_path

def close2zero(joints: pb2.JointPosition, threshold: float = 0.01) -> bool:
    return (abs(joints.J1) + abs(joints.J2) + abs(joints.J3) + abs(joints.J4) + abs(joints.J5) + abs(joints.J6)) < threshold
        
joints1 = pb2.JointPosition(
J1=5.235987755982989e-05,
J2=-1.7453292519943296e-05,
J3=-0.3491007569839058,
J4=-1.7453292519943296e-05,
J5=0.3490658503988659,
J6=0
)

joints2 = pb2.JointPosition(
  J1 = 0.6083345107826235,
  J2=-0.2557430952947291,
  J3=-0.21579250871657893,
  J4=0.7409446273491528,
  J5=0.8065864605166594,
  J6=-1.1503290566969429
)

joints3=  pb2.JointPosition(
      J1=0.5699547205312683,
  J2=-0.3974463772641487,
  J3=-0.26124088243851123,
  J4=0.5735675520828966,
  J5=0.8849517439312049,
  J6=-1.2509298347818958,
)


joints4 = pb2.JointPosition(
  J1=0.5039987280984025,
  J2=-0.5222548720742632,
  J3=-0.06733480254194124,
  J4= 0.5264436622790496,
  J5= 0.7980518004744072,
  J6= -1.2381365713647774
)
joints5 = pb2.JointPosition(
  J1=0.49801224876406197,
  J2= -0.6660874557311159,
  J3= 0.17385224679115518,
  J4= 0.917868653623818,
  J5= 0.8937132967762165,
  J6= -1.5856141721443284
)
joints6 =  pb2.JointPosition(
  J1= 0.4199436713223556,
  J2= -0.605716516904632,
  J3= 0.08740608893987602,
  J4= 0.904621604601181,
  J5= 0.828560155799268,
  J6= -1.2329180369013144
)
joints7 = pb2.JointPosition(
  J1= 0.27164304478039747,
  J2= -0.4979075290089423,
  J3= 0.018046704465621368,
  J4= 0.8484220026869634,
  J5= 0.6629807696625659,
  J6= -0.9022828634035086
)
joints8 =  pb2.JointPosition(
  J1=-0.09911724822075799,
  J2= -0.4549375228248419,
  J3= -0.05311036913818745,
  J4= -0.37124898519171384,
  J5= 0.5555383009097951,
  J6= 0.3500432347799827
)
joints9 = pb2.JointPosition(
  J1= -0.3816511475336001,
  J2= -0.5456073774659473,
  J3= 0.08806931405563387,
  J4= -0.8433430945636601,
  J5= 0.6590363255530588,
  J6= 0.7618711250805648
)
joints10 = pb2.JointPosition(
  J1= -0.6028192703463215,
  J2= -0.6472029932245373,
  J3= 0.24516640002764348,
  J4= -1.0624691821515482,
  J5= 0.8061501282036608,
  J6= 1.3163273218541234,
)
joints11 =  pb2.JointPosition(
  J1= -0.878755825086625,
  J2= -0.2697231826032037,
  J3= -0.32695252877609776,
  J4= -1.1504163231595423,
  J5= 1.1096977917105146,
  J6= 1.2382063845348572
)

trajectory_joints = [joints1, joints2, joints3, joints4, joints5, joints6, joints7, joints8, joints9, joints10, joints11]

def run():
    with grpc.insecure_channel("0.0.0.0:49999") as channel:
        algorithm_stub = pb2_grpc.ControllerServerStub(channel)
        algorithm_version = algorithm_stub.getServerVersion(pb2.FakeInput())
        print("algorithm_version result: {}".format(algorithm_version))
        # test 2 rounds of most interfaces in grpc
        for i in range(2):
            # connect
            connect_result = algorithm_stub.connect(pb2.ControllerIP(controller_ip= "192.168.110.2"))
            print("connect result: {}".format(connect_result))
            # reset 
            reset_result = algorithm_stub.reset(pb2.FakeInput())
            print("reset result: {}".format(reset_result))
            # setup
            setup_result = algorithm_stub.setUpRobot(pb2.FakeInput())
            print("setup result: {}".format(setup_result))
            # get name
            getname_result = algorithm_stub.getName(pb2.FakeInput())
            print("getName result: {}".format(getname_result))
            # set speed ratio
            setSpeedRatio_result = algorithm_stub.setSpeedRatio(pb2.SetSpeedRatioRequest(speed_ratio=i * 0.4 + 0.1))
            print("setSppedRatio result: {}".format(setSpeedRatio_result))
            # set IO
            io_states = [pb2.IOState(io=5, state=i, io_type=2),
                         pb2.IOState(io=6, state=i, io_type=2),
                         pb2.IOState(io=7, state=i, io_type=2),
                         pb2.IOState(io=8, state=i, io_type=2)]
            setIO_result = algorithm_stub.setIO(pb2.SetIORequest(io_states = io_states))
            print("setIO result: {}".format(setIO_result))
            # get IO
            getIO_result = algorithm_stub.getIO(pb2.GetIORequest(io_channels=[1,2,3,4,5,6,7,8], io_type=2))
            print("getIO result: {}".format(getIO_result))
            # servo on
            enable_result = algorithm_stub.enableRobot(pb2.FakeInput())
            print("enable resul: {}".format(enable_result))
            # get joint position
            getJointPose_result: pb2.GetJointPositionReply = algorithm_stub.getJointPosition(pb2.FakeInput())
            current_joint_pose = getJointPose_result.joints
            print("getJointPose result: {}".format(getJointPose_result))
            # move joint
            if close2zero(current_joint_pose):
                target_joint_pose = pb2.JointPosition(J1=0, J2=0, J3=math.radians(20), J4=0, J5=0, J6=0)
            else:
                target_joint_pose = pb2.JointPosition(J1=0, J2=0, J3=0, J4=0, J5=0, J6=0)
            jointMove_result = algorithm_stub.moveJoint(target_joint_pose)
            print("jointMove result: {}",format(jointMove_result))
            # get cartesian position
            getPose_result: pb2.GetPoseReply = algorithm_stub.getPose(pb2.FakeInput())
            current_cartesian_pose = getPose_result.pose
            print("getPose result: {}".format(getPose_result))
            # move pose 
            target_cartesian_pose = pb2.CartesianPosition(x=current_cartesian_pose.x, y=current_cartesian_pose.y + 100, z=current_cartesian_pose.z,
                                                          rx=current_cartesian_pose.rx, ry=current_cartesian_pose.ry, rz=current_cartesian_pose.rz)
            cartesian_move_result = algorithm_stub.movePose(target_cartesian_pose)
            print("cartesianMove result: {}",format(cartesian_move_result))
            # test uploadFlyShotTraj
            shot_flags = [True for _ in trajectory_joints]
            shot_flags[0] = False
            one_io = pb2.IOAddrs(addrs=[1,2,3,4,5,6,7,8])
            trajectory_addres = [ one_io for _ in trajectory_joints ]
            
            upload_request = pb2.UploadFlyShotTrajRequest(name="unit_test_trajectory", 
                                                          joints=trajectory_joints,
                                                          shot_flags=shot_flags,
                                                          offset=[0 for _ in trajectory_joints],
                                                          traj_addrs=trajectory_addres)
            upload_result = algorithm_stub.uploadFlyShotTraj(upload_request)
            print("upload result: {}",format(upload_result))
            # optimizeKinematicsParams
            kinematics_result: pb2.OptimizeKinematicsParamsReply = algorithm_stub.optimizeKinematicsParams(pb2.OptimizeKinematicsParamsRequest(name="unit_test_trajectory", method="nothing"))
            kinematics = kinematics_result.kinematics_params
            print("kinematics result: {}",format(kinematics_result))
            # is FlyShotTrajValid
            valid_result = algorithm_stub.isFlyShotTrajValid(pb2.IsFlyShotTrajValidRequest(name="unit_test_trajectory", 
                                                                                           method="TOTP",
                                                                                           save_traj=True,
                                                                                           kinematics_params=kinematics))
            print("valid result: {}",format(valid_result))
            # execute trajectory
            execute_trajectory_result = algorithm_stub.executeFlyShotTraj(pb2.ExecuteFlyShotTrajRequest(name="unit_test_trajectory",
                                                                                                       method="nothing",
                                                                                                       move_to_start=True,
                                                                                                       use_cache=False,
                                                                                                       kinematics_params=kinematics))
            print("execute_trajectory_result: {}".format(execute_trajectory_result))
            # servo off
            disable_result = algorithm_stub.disableRobot(pb2.FakeInput())
            print("disable resul: {}".format(disable_result))
            # delete trajectory
            delete_result = algorithm_stub.deleteFlyShotTraj(pb2.DeleteFlyShotTrajRequest(name="unit_test_trajectory"))
            print("delete resul: {}".format(delete_result))

if __name__ == "__main__":
    logging.basicConfig()
    run()