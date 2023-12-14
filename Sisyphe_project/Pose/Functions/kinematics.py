import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
import numpy as np 
from ForwardInverseKinematics.RobotForwardKinematics import RobotForwardKinematics
from Pose.utils.home_joints import HomeJoints
from Pose.utils.tools import real2virtual, virtual2real, euler_to_homogeneous_matrix, homogeneous_matrix_to_euler
from ForwardInverseKinematics.RobotinverseKinematics import RobotinverseKinematics
from typing import Union

def robotForwardKinematics(thetas:np.ndarray, manipulator_type: str) -> np.ndarray:
    """Forward kinematics of robot.

    Args:
        thetas (np.ndarray): The angle of the joint axis of the robotic arm in a real environment. unit in degrees.
        manipulator_type (str): Type of robot. 

    Returns:
        xyzrxryrz (np.ndarray): The pose of the robotic arm's end effector in workspace. Units in 
        millimeters and degrees.
    """

    thetas = real2virtual(thetas, manipulator_type)
    FK = RobotForwardKinematics(manipulator_type)
    trans_matrix = FK.forward_kinematics(thetas)
    xyzrxryrz = homogeneous_matrix_to_euler(trans_matrix)

    return xyzrxryrz

def robotInverseKinematics(xyzrxryrz:np.ndarray, start_thetas:np.ndarray, manipulator_type: str) -> Union[np.ndarray, bool]:
    """Inverse kinematics of robot.

    Args:
        xyzrxryrz (np.ndarray): The pose of the robotic arm's end effector in workspace. Units in 
        millimeters and degrees.
        start_thetas (np.ndarray): Reference angle of the joint axis of the robotic arm in a real environment. unit in degrees.
        manipulator_type (str): Type of robot. 

    Returns:
        inverse_thetas (np.ndarray): The angle of the joint axis of the robotic arm in a real environment. unit in degrees.
        is_qualified (bool):
    """
    
    if start_thetas is None:
        start_thetas = HomeJoints
    start_thetas = real2virtual(start_thetas, manipulator_type)
    target_matrix = euler_to_homogeneous_matrix(xyzrxryrz)
    IK = RobotinverseKinematics(manipulator_type)
    inverse_thetas, is_qualified = IK.inverse_kinematics(target_matrix, start_thetas)
    # print(f"in{inverse_thetas}")
    if is_qualified:
        inverse_thetas = virtual2real(inverse_thetas, manipulator_type)
    
    return inverse_thetas, is_qualified



