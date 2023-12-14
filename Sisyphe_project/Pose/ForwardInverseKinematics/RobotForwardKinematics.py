from typing import List, Union
import numpy as np
from math import cos, sin
import copy
from .robotParameters import RobotParameters
# from .robot_Parameter import RobotParameters




class RobotForwardKinematics():
    def __init__(self, robot_manipulator_type: str) -> None:
        self.__robot_manipulator_type: str = robot_manipulator_type
        self.__dhp:np.ndarray = RobotParameters[self.__robot_manipulator_type]["DHP"]

    @staticmethod
    def Tran_matrix_A(a: float, alpha: float, d: float, theta: float) -> np.array:
        """计算机械臂坐标系的转换矩阵
        Args:
            a: 沿着当前x轴的平移
            alpha: 绕当前x轴的旋转
            d: 沿着当前z轴的平移
            theta: 绕当前z轴的旋转
        Returns:
            齐次矩阵
        
        Rot_theta = np.array([
            [cos(theta), -sin(theta), 0, 0],
            [sin(theta),  cos(theta), 0, 0],
            [         0,           0, 1, 0],
            [         0,           0, 0, 1]
        ])
        Trans_d = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, d],
            [0, 0, 0, 1]
        ])
        Trans_a = np.array([
            [1, 0, 0, a],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        Rot_alpha = np.array([
            [1,          0,           0, 0],
            [0, cos(alpha), -sin(alpha), 0],
            [0, sin(alpha),  cos(alpha), 0],
            [0,          0,           0, 1]
        ])
        
        mat_A_ = Rot_theta @ Trans_d @ Trans_a @ Rot_alpha
        """
        
        mat_A = np.array([
            [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
            [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
            [0         ,  sin(alpha)           ,  cos(alpha)           , d           ],
            [0         ,  0                    , 0                     , 1           ]
        ])
        
        return mat_A

    def forward_kinematics(self, joints: np.ndarray, all_transformations: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
        
        forward_para_list = copy.deepcopy(self.__dhp)
        forward_para_list[1:,3] = forward_para_list[1:,3] + joints
        trans_matrix = np.eye(4)
        trans_matrix_list = []
        for para in forward_para_list:
            trans_matrix = trans_matrix @ self.Tran_matrix_A(*para)
            trans_matrix_list.append(trans_matrix)
        if all_transformations:
            return trans_matrix_list
        else:
            return trans_matrix_list[-1]
    

    