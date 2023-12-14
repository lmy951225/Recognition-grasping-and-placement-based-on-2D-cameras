'''
Author      :XiaoZhiheng
Time        :2023/06/13 15:57:32
'''

from .RobotAnalyticalInverseKinematics import RobotAnalyticalInverseKinematics
from .RobotLevenbergMarquardtInverseKinematics import RobotLevenbergMarquardtInverseKinematics
from Pose.utils.paramaters_utils import IKSolverType
import numpy as np 


class RobotinverseKinematics():
    
    def __init__(self, robot_manipulator_type) -> None:
        
        if IKSolverType == "LM":
            self.IKSolver = RobotLevenbergMarquardtInverseKinematics(robot_manipulator_type)
        elif IKSolverType == "AN":
            self.IKSolver = RobotAnalyticalInverseKinematics(robot_manipulator_type)
        else:
            raise ValueError("Unkonwn IKSolverType: {}".format(IKSolverType))
        
        # print("IKSolverType is", IKSolverType)
        
    def inverse_kinematics(self, target_matrix: np.ndarray, start_thetas: np.ndarray):
    
        return self.IKSolver.inverse_kinematics(target_matrix, start_thetas) 
            
    @property
    def upper_boundary(self) -> np.ndarray:
        return self.IKSolver.upper_boundary
    
    @property
    def lower_boundary(self) -> np.ndarray:
        return self.IKSolver.lower_boundary
    
    
    

