"""
@Author      :   XiaoZhiheng
@Time        :   2023/02/08 11:11:21
"""

import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
import numpy as np 
from math import pi, sqrt, sin, acos, atan, cos
from typing import List, Tuple
from Pose.utils.tools import euler_to_homogeneous_matrix
from copy import deepcopy
import logging
from ForwardInverseKinematics.robotParameters import RobotParameters
from ForwardInverseKinematics.SuperRobotInverseKinematics import SuperRobotInverseKinematics
from ForwardInverseKinematics.RobotForwardKinematics import RobotForwardKinematics

class RobotAnalyticalInverseKinematics(SuperRobotInverseKinematics):

    def __init__(self, robot_manipulator_type) -> None:

        super().__init__(robot_manipulator_type)
        self.dhp = RobotParameters[robot_manipulator_type]["DHP"]
        self.FK = RobotForwardKinematics(robot_manipulator_type)

    def inverse_kinematics(self, target_matrix: np.ndarray, start_thetas: np.ndarray):
        
        # print("target_matrix is:", target_matrix)
        all_inverse_thetas = self.analystic_IK(target_matrix)
        is_qualified = False
        inverse_thetas = None
        dist = float("inf")
        dist_list = []
        qualify_list = []
        alternative_list = []

        for inverse_thetas in all_inverse_thetas:
            M = self.FK.forward_kinematics(inverse_thetas)
            if np.max(np.abs(M-target_matrix)) > 0.001:
                continue
            inverse_thetas, is_qualified = self.angle_check_and_modified(inverse_thetas)
            if is_qualified:
                dist = self.angle_dist(inverse_thetas, start_thetas)
            else:
                dist = float("inf")
            dist_list.append(dist)
            qualify_list.append(is_qualified)
            alternative_list.append(inverse_thetas)
        if dist_list == []:
            is_qualified = False
            inverse_thetas = None
            min_dist = float("inf")
        else:
            min_dist = min(dist_list)
            min_index = dist_list.index(min_dist)
            is_qualified = qualify_list[min_index]
            inverse_thetas = alternative_list[min_index]
        logging.info("The final inverse_thetas is:{}; is_qualified is {}; dist to start points"
                              " is {} (radians).".format(inverse_thetas, is_qualified, min_dist))
        
        return inverse_thetas, is_qualified

    @staticmethod
    def function_solver(A:float, B:float, C:float)-> Tuple[List]:
        
        """
        solve_function
        A * sin(a) + B*cos(a)=C 
        """
        a = A**2 +  B**2
        b = -2*A*C
        c = C**2 - B**2
        detal = b ** 2 - 4 * a*c
        if detal < 0:
            return [], []
        elif detal == 0:
            sin_a = -b / (2*a)
            cos_a = (C-A * sin_a) / B
            return [sin_a], [cos_a]
        else:
            sin_a_1 = (-b+sqrt(detal)) /(2*a)
            cos_a_1 = (C-A * sin_a_1) / B
            sin_a_2 = (-b-sqrt(detal)) /(2*a)
            cos_a_2 = (C-A * sin_a_2) / B
            
            return [sin_a_1, sin_a_2], [cos_a_1, cos_a_2]
    
    @staticmethod
    def function_solver_sincos(sin_A: float, cos_A:float) -> float:
        """solve function as sin(A)=sin_A, cos(A)=cos_A
        Args:
            sin_A (float): Value of sin(A)
            cos_A (float): Value of sin(A)
        Returns:
            float: A 
        """
        
        theta = atan(sin_A / cos_A)
        
        e = 1e-8
        if cos_A > e:
            A = theta 
        elif sin_A > 0 and cos_A < -e:
            A = theta + pi
        elif sin_A < 0 and cos_A < -e:
            A = theta - pi
        elif abs(cos_A) < e and sin_A > 0:
            A = pi/2
        elif abs(cos_A) > e and sin_A < 0:
            A = -pi/2    
            
        return A    
        
    def analystic_IK(self, target_Tran_matrix:np.ndarray) -> List:

        target_R = target_Tran_matrix[:3, :3]
        target_o = target_Tran_matrix[:3, 3]
        d6 = self.dhp[6, 2]
        a = list(self.dhp[:,0])
        d = list(self.dhp[:,2])
        point_o4 = target_o - d6 * target_R @ np.array([0 , 0, 1])
        x4, y4, z4 = point_o4
        theta1 = atan(y4/x4) 
        s = z4 - d[1] -d[0]
        r = y4 / sin(theta1) - a[1]
        sin_theta3_list, cos_theta3_list = self.function_solver(-2*a[2]*d[4], 2*a[2]*a[3], (r**2+s**2-a[2]**2-a[3]**2-d[4]**2))
        theta2_list = []
        theta3_list = []
        
        for sin_theta3, cos_theta3 in zip(sin_theta3_list, cos_theta3_list):   
            sin_theta2_list, cos_theta2_list = self.function_solver(-(a[3] * sin_theta3+d[4]*cos_theta3), a[2]+a[3]*cos_theta3-d[4]*sin_theta3, r)
            for sin_theta2, cos_theta2 in zip(sin_theta2_list, cos_theta2_list):
                sin_theta23 = sin_theta2*cos_theta3 + cos_theta2*sin_theta3
                cos_theta23 = cos_theta2*cos_theta3 - sin_theta2*sin_theta3
                s_ = -a[2] * sin_theta2 - a[3] * sin_theta23 -d[4] * cos_theta23
                if s_- s < 1e-5:
                    theta2 = self.function_solver_sincos(sin_theta2, cos_theta2) 
                    theta3 = self.function_solver_sincos(sin_theta3, cos_theta3)
                    theta2_list.append(theta2)
                    theta3_list.append(theta3)
        all_inverse_thetas = []
        for theta2, theta3 in zip(theta2_list, theta3_list):
            R_03 = np.array([
                [cos(theta1)*cos(theta2)*cos(theta3)-cos(theta1)*sin(theta2)*sin(theta3), sin(theta1), -cos(theta1)*cos(theta2)*sin(theta3)-cos(theta1)*sin(theta2)*cos(theta3)],
                [sin(theta1)*cos(theta2)*cos(theta3)-sin(theta1)*sin(theta2)*sin(theta3),-cos(theta1), -sin(theta1)*cos(theta2)*sin(theta3)-sin(theta1)*sin(theta2)*cos(theta3)],
                [-sin(theta2)*cos(theta3)-cos(theta2)*sin(theta3), 0, sin(theta2)*sin(theta3)-cos(theta2)*cos(theta3)]
            ])
            R_36 = R_03.T @ target_R
            theta5 = acos(R_36[2,2])
            theta5_list = [theta5, -theta5]
            e = 1e-8
            
            inverse_thetas = np.zeros(6)
            for theta5 in theta5_list:
                sin_theta5 = sin(theta5)
                if sin_theta5 > e:
                    theta6 = self.function_solver_sincos(-R_36[2,1], R_36[2,0]) 
                    theta4 = self.function_solver_sincos(-R_36[1,2], -R_36[0,2])
                elif sin_theta5 < -e:
                    theta6 = self.function_solver_sincos(R_36[2,1], -R_36[2,0]) 
                    theta4 = self.function_solver_sincos(R_36[1,2], R_36[0,2])
                elif abs(sin_theta5) <= e and R_36[2,2] > 0:
                    theta46 = self.function_solver_sincos(R_36[1,0], R_36[0,0])
                    theta4 = theta46 / 2
                    theta6 = theta46 - theta4
                elif abs(sin_theta5) <= e and R_36[2,2] < 0:
                    break 
                inverse_thetas = np.array([theta1, theta2 + pi/2, theta3, theta4, theta5, theta6])
                all_inverse_thetas.append(inverse_thetas)
                
        return all_inverse_thetas
    
    
if __name__ == "__main__":
    # np.set_printoptions(suppress=True, precision=6)
    
    from utils.tools import virtual2real, real2virtual, euler_to_homogeneous_matrix
    from utils.home_joints import HomeJoints
    import time

    AIK_solver = RobotAnalyticalInverseKinematics("LR_Mate_200iD_7L")

    xyzrxryrz = np.array([448.465286960615,	-380.465789798864,	627.648485754523,	
     -96.68501762787,	81.889753559655,	-4.55171812112072])
    start_thetas = HomeJoints
    start_thetas = real2virtual(start_thetas, "LR_Mate_200iD_7L", isToRadians=True)
    target_matrix = euler_to_homogeneous_matrix(xyzrxryrz)
    t1 = time.time()
    thetas, is_q = AIK_solver.inverse_kinematics(target_matrix, start_thetas)
    t2 = time.time()
    # all_inverse_thetas = np.array(all_inverse_thetas)
    # all_inverse_thetas = [for ]
    print(t2-t1)
    print(virtual2real(thetas, "LR_Mate_200iD_7L", isToAngle=True), is_q)
    
