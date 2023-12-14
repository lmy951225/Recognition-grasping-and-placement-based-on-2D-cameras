"""
@Author      :   XiaoZhiheng
@Time        :   2023/02/13 18:40:45
"""

import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from typing import List, Tuple, Union
from math import pi, sqrt, sin, acos, tan 
import numpy as np
from Pose.utils.tools import euler_to_homogeneous_matrix, transform_xyzrxryrz
from ForwardInverseKinematics.SuperRobotInverseKinematics import SuperRobotInverseKinematics
import logging
from copy import deepcopy

class RobotLevenbergMarquardtInverseKinematics(SuperRobotInverseKinematics):
    
    def __init__(self, robot_manipulator_type: str) -> None:
        
        super().__init__(robot_manipulator_type)
        self.max_interation_num = 250
        self.random_try_num = 10
        np.random.seed(441700)

    def inverse_kinematics(self, target_matrix: np.ndarray, start_thetas: np.ndarray) -> Tuple[np.ndarray, bool]:
        
        # xyzrxryrz = deepcopy(target_matrix)
        # xyzrxryrz = transform_xyzrxryrz(target_matrix)
        # target_matrix = euler_to_homogeneous_matrix(xyzrxryrz) 
        inverse_thetas = self.levenberg_marquardt_IK(start_thetas, target_matrix)
        if inverse_thetas is not None:
            inverse_thetas, is_qualified = self.angle_check_and_modified(inverse_thetas)
        else:
            is_qualified = False
        is_angle_too_large = False
        dist = float("inf")
        if is_qualified:
            dist = self.angle_dist(start_thetas, inverse_thetas)
            is_angle_too_large = dist > 3.14
        
        alternative_list = [inverse_thetas]
        dist_list = [dist]
        qualify_list = [is_qualified]
        if not is_qualified or is_angle_too_large:
            for i in range(self.random_try_num):
                inverse_thetas = self.levenberg_marquardt_IK(self.random_angle, target_matrix)
                if inverse_thetas is not None:
                    inverse_thetas, is_qualified = self.angle_check_and_modified(inverse_thetas)
                else:
                    is_qualified = False
                if is_qualified:
                    dist = self.angle_dist(start_thetas, inverse_thetas)
                else:
                    dist = float("inf")
                dist_list.append(dist)
                alternative_list.append(inverse_thetas)
                qualify_list.append(is_qualified)
        min_index = dist_list.index(min(dist_list))
        is_qualified = qualify_list[min_index]
        inverse_thetas = alternative_list[min_index]
            
        logging.info("The final inverse_thetas is:{}; is_qualified is {}; dist to start points"
                            " is {} (radians).".format(inverse_thetas, is_qualified, min(dist_list)))
        return inverse_thetas, is_qualified

    def levenberg_marquardt_IK(
        self, current_thetas: np.ndarray, desired_pose: np.ndarray, tol: float = 1e-4, 
        damping: float = 0.04, max_rate: float = 0.032
    ) -> Union[np.ndarray, None]:

        trans_matrix_list = self.FK.forward_kinematics(current_thetas, True)
        current_pose = trans_matrix_list[-1]
        Jacobian_matrix = self.compute_jacobian_matrix(trans_matrix_list)
        for _ in range(self.max_interation_num):
            step_twist = self.get_desired_twist(current_pose, desired_pose)
            dtheta = Jacobian_matrix.T @ np.linalg.inv(Jacobian_matrix @ Jacobian_matrix.T + damping ** 2 * np.identity(6)) @ step_twist
            rate = max_rate / max(max_rate, np.max(np.abs(dtheta)))
            next_thetas = current_thetas + rate * dtheta
            trans_matrix_list = self.FK.forward_kinematics(next_thetas, True)
            next_pose = trans_matrix_list[-1]
            Jacobian_matrix = self.compute_jacobian_matrix(trans_matrix_list)
            current_thetas = next_thetas
            current_pose = next_pose
            delta = np.linalg.norm(current_pose - desired_pose)
            if delta < tol:
                break
        if delta > tol:
            return None
        else:
            return next_thetas
          

    @staticmethod
    def compute_jacobian_matrix(trans_matrix_list: List[np.ndarray]) -> np.ndarray:

        transforms = trans_matrix_list
        end_pos = transforms[-1][0:3, 3]
        transforms = [np.identity(4)] + transforms[1:-1]
        jacobian_mat = np.zeros((len(transforms), len(transforms)))
        for i, Tran in enumerate(transforms):
            z_axis = Tran[0:3, 2]
            joint_pos = Tran[0:3, 3]
            joint_to_end = np.cross(z_axis, end_pos - joint_pos)
            jacobian_mat[0:3, i] = joint_to_end
            jacobian_mat[3:6, i] = z_axis
     
        return jacobian_mat
    
    @property
    def random_angle(self) -> np.ndarray:

        return np.random.uniform(self.lower_boundary, self.upper_boundary, 6)

    def get_desired_twist(self, current: np.ndarray, desired: np.ndarray) -> np.ndarray:
        s = self.trlog_(self.trnorm_(desired @ np.linalg.inv(current)))
        if np.isnan(s).any():
            s = np.zeros(6)
            s[0:3] = desired[0:3, 3] - current[0:3, 3]
            s[3:6] = 0
        return s

    @staticmethod
    def trnorm_(M: np.ndarray) -> np.ndarray:

        def unitvec_(vec: np.ndarray) -> np.ndarray:

            return vec / sqrt(sum(vec * vec))

        o = M[:3, 1]
        a = M[:3, 2]
        n = np.cross(o, a)  # N = O x A
        o = np.cross(a, n)
        R = np.stack((unitvec_(n), unitvec_(o), unitvec_(a)), axis=1)
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = M[:3,3]
     
        return T

    @staticmethod
    def trlog_(T: np.ndarray) -> np.ndarray:

        def iseye(T: np.ndarray) -> bool:
            return np.linalg.norm(T - np.eye(T.shape[0])) < 1e-8

        if iseye(T):
            return np.zeros((6,))
        else:
            R = T[:3,:3]
            t = T[:3,3]
            if iseye(R):
                return np.r_[t, [0, 0, 0]]
            else: 
                if abs(np.trace(R) + 1) < 100 * np.finfo(np.float64).eps:
                    diagonal = R.diagonal()
                    k = diagonal.argmax() 
                    mx = diagonal[k]
                    I = np.eye(3)
                    col = R[:, k] + I[:, k]
                    w = col / np.sqrt(2 * (1 + mx))
                    theta = pi 
                    v = w * theta
                    S = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                else:
                    theta = acos((np.trace(R) - 1) / 2)
                    sin_theta = sin(theta)
                    if sin_theta == 0:
                        S = np.zeros((3,3))
                    else:
                        skw = (R - R.T) / 2 / sin(theta)
                        S = skw * theta
                w = np.array([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0], S[1, 0] - S[0, 1]]) / 2                      
                theta = sqrt(sum(w * w))
                if theta == 0:
                    return np.r_[t, 0, 0, 0]
                else:
                    Ginv = np.eye(3) - S / 2 + (1 / theta - 1 / tan(theta / 2) / 2) / theta * S @ S
                    v = Ginv @ t
                    return np.r_[v, w]




if __name__ == "__main__":
    
    from utils.tools import virtual2real, real2virtual, euler_to_homogeneous_matrix
    from utils.home_joints import HomeJoints  

    LMIK = RobotLevenbergMarquardtInverseKinematics("LR_Mate_200iD_7L")
    
    
    xyzrxryrz = np.array([448.465286960615,	-380.465789798864,	627.648485754523,	
     -96.68501762787,	81.889753559655,	-4.55171812112072])
    start_thetas = HomeJoints
    target_matrix = euler_to_homogeneous_matrix(xyzrxryrz)
    start_thetas = real2virtual(start_thetas, "LR_Mate_200iD_7L", isToRadians=True)
    thetas, is_q = LMIK.inverse_kinematics(target_matrix, start_thetas)

    
    print(virtual2real(thetas, "LR_Mate_200iD_7L", isToAngle=True))

