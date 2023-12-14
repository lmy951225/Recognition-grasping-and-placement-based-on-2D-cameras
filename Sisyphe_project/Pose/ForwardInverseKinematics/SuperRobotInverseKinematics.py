import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from abc import abstractmethod
from typing import List, Tuple
from math import pi 
import numpy as np
from ForwardInverseKinematics.robotParameters import RobotParameters
from ForwardInverseKinematics.RobotForwardKinematics import RobotForwardKinematics
from Pose.utils.tools import real2virtual

class SuperRobotInverseKinematics():
    def __init__(self, robot_manipulator_type: str) -> None:

        self._robot_manipulator_type = robot_manipulator_type
        self.__lower_boundary = RobotParameters[self._robot_manipulator_type]['LowerBoundary']
        self.__upper_boundary = RobotParameters[self._robot_manipulator_type]['UpperBoundary']

        virtual_boundary = np.array([real2virtual(self.__lower_boundary, self._robot_manipulator_type), 
                                     real2virtual(self.__upper_boundary, self._robot_manipulator_type)])
        self.lower_boundary = np.min(virtual_boundary, axis=0)
        self.upper_boundary = np.max(virtual_boundary, axis=0)
        self.FK = RobotForwardKinematics(self._robot_manipulator_type)

    # @staticmethod
    # def transform_xyzrxryrz(xyzrxryrz: np.ndarray) -> np.ndarray: 

    #     xyzrxryrz[:3] = xyzrxryrz[:3] / 1000
    #     xyzrxryrz[3:] = xyzrxryrz[3:] / 180 * pi

    #     return xyzrxryrz

    
    @abstractmethod
    def inverse_kinematics(self, target_matrix: np.ndarray, start_theta: np.ndarray) -> Tuple[List[float], bool]:
        raise NotImplementedError
    
    def angle_check_and_modified(self, thetas: np.ndarray) -> Tuple[np.ndarray, bool]:

        new_thetas = np.zeros_like(thetas) 
        is_qualified = True
        for i, (theta, upper_b, lower_b) in enumerate(zip(thetas, self.upper_boundary, self.lower_boundary)):
            theta = theta - int(theta / (2*pi)) *2*pi
            if theta > upper_b:
                theta = theta - 2*pi
            elif theta < lower_b:
                theta = theta + 2*pi
            else:
                theta = theta
                
            if theta > upper_b or theta < lower_b:
                is_qualified = False
            new_thetas[i] = theta

        return new_thetas, is_qualified
    
    @staticmethod
    def angle_dist(angle_A: np.ndarray, angle_B: np.ndarray) -> float:

        weights = np.array([1.6, 1.6, 1.6, 2.0, 1.6, 1.6])  * 0.6
       
        return sum(weights * (angle_A - angle_B) ** 2 ) ** 0.5