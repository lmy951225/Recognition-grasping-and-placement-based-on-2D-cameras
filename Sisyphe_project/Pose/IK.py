import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)

import numpy as np 
from Pose.utils.tools import read_json,virtual2real
from Functions.kinematics import robotInverseKinematics
from math import pi 
# from utils.path_util import parameters_path, obstacles_config_path
from Pose.utils.path_util import get_parameters_path
from Pose.utils.home_joints import HomeJoints


class getJoint():
    def __init__(self) -> None:
        super().__init__()
        self.__parameters = read_json(get_parameters_path())
        self.__manipulator_type = self.__parameters["ManipulatorType"]

    def transferCartesian2Joints(self,pose):
        xyzrxryrz = np.array([pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]])
        
        inverse_thetas, is_qualified = robotInverseKinematics(xyzrxryrz,HomeJoints,self.__manipulator_type)
        # print(f'xyz{xyzrxryrz}--theta{HomeJoints}---type{self.__manipulator_type}')
        # print(f'iners_theta{inverse_thetas}---is_qu{is_qualified}')
        # inverse_thetas, is_qualified = robotLevenbergMarquardtInverseKinematics(xyzrxryrz,self.__manipulator_type)
        if is_qualified:
            inverse_thetas = inverse_thetas / 180 * pi
            # inverse_thetas = virtual2real(inverse_thetas, self.__manipulator_type, isToAngle=False)
            # print(f'tehta{inverse_thetas}')
            joints = np.array([inverse_thetas[0],inverse_thetas[1],inverse_thetas[2],inverse_thetas[3],inverse_thetas[4],inverse_thetas[5]])
        else:
            # print(f"none")
            joints = np.array([-99999,-99999,-99999,-99999,-99999,-99999])
            
        return joints 