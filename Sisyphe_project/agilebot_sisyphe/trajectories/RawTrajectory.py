from typing import Dict, Tuple, List, Union
import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
import json
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import toppra
from trajectories.Trajectory import Trajectory
from trajectories.MetaTrajectoryTOPPRA import MetaTrajectoryTOPPRA
from utils.parameter_util import POSITION_CONSTRAINT, VELOCITY_CONSTRAINT, ACCELERATION_CONSTRAINTS, JERK_CONSTRAINTS, TIME_STEP

def permute(nums):
    
    def backtrack(first = 0):
        # all number finish
        if first == n:
            res.append(nums[:])
        for i in range(first, n):
            nums[first], nums[i] = nums[i], nums[first]
            backtrack(first=first + 1)
            nums[first], nums[i] = nums[i], nums[first]
    
    n = len(nums)
    res = []
    backtrack()
    return res

five_joints_orders = permute([1, 2, 3, 4, 5])


NUM_POINTS_RATIO = 3

class RawTrajectory(Trajectory):
    def __init__(self, trajectory_name: str, trajectory_path: str) -> None:
        super().__init__(trajectory_name, trajectory_path)
        with open(trajectory_path, "r") as f:
            trajectory_content = json.load(f)
        self.__way_points = trajectory_content["flying_shots"][trajectory_name]["traj_waypoints"]# only working for shot_flags set
        self.__transition_way_points = trajectory_content["flying_shots"][trajectory_name]["trajectory_transition_waypoints"]# real waypoints for toppra
        self.__trajectory_transition_states:List[Union[int, str]] = trajectory_content["flying_shots"][trajectory_name]["trajectory_transition_states"]
        self.__shot_flags = trajectory_content["flying_shots"][trajectory_name]["shot_flags"]
        self.__io_addr_info = trajectory_content["flying_shots"][trajectory_name]["addr"]
        self.__trajectory_len = len(self.__shot_flags)
        self.__transition_trajectory_len = len(self.__transition_way_points)
        self.__spline_path = toppra.SplineInterpolator(np.linspace(0, 1, self.__transition_trajectory_len), self.__transition_way_points, bc_type="natural")
        self.__grid_points = np.linspace(0, self.__spline_path.duration, NUM_POINTS_RATIO * (self.__transition_trajectory_len - 1) + 1)
    
    @property
    def way_points(self) -> List[List[float]]:
        return self.__way_points

    @property
    def shot_flags(self) -> List[bool]:
        return self.__shot_flags
    
    @property
    def trajectory_len(self) -> int:
        return self.__trajectory_len
    
    @property
    def first_point(self) -> List[float]:
        return self.__way_points[0]
    
    @property
    def last_point(self) -> List[float]:
        return self.__way_points[-1]
    
    @property
    def trajectory_transition_states(self) -> List[Union[int, str]]:
        return self.__trajectory_transition_states

    def __search_joint_greedy(self, joint_index: int, pc_velocity: toppra.constraint.JointVelocityConstraint, acceleration_factors: np.ndarray) -> Tuple[np.ndarray, float]:
        acceleration_factor_max = 100
        acceleration_factor_min = 1
        while acceleration_factor_min < acceleration_factor_max:
            acceleration_factor_mid = int((acceleration_factor_max + acceleration_factor_min) / 2)
            acceleration_factors[joint_index] = acceleration_factor_mid * 0.01
            pc_acceleration = toppra.constraint.JointAccelerationConstraint(acceleration_factors * ACCELERATION_CONSTRAINTS)
            toppra_instance = toppra.algorithm.TOPPRA(constraint_list=[pc_velocity, pc_acceleration],
                                                    path=self.__spline_path, solver_wrapper="seidel", 
                                                    gridpoints=self.__grid_points, parametrizer="ParametrizeSpline")
            meta_trajectory = MetaTrajectoryTOPPRA(toppra_instance=toppra_instance)
            exceed_result = meta_trajectory.check_all_limits(check_flags=[False, True, True, False], joint_index=joint_index)
            if exceed_result == -1:
                duration = meta_trajectory.duration
                acceleration_factor_min = acceleration_factor_mid + 1
            else:
                acceleration_factor_max = acceleration_factor_mid

        if exceed_result != -1:
            acceleration_factors[joint_index] -= 0.01
        
        return acceleration_factors, duration
    
    def __testify_all_joints(self, pc_velocity: toppra.constraint.JointVelocityConstraint, acceleration_factors: np.ndarray) -> Tuple[np.ndarray, float]:
        search_time = 100
        while search_time > 0:
            search_time -= 1
            pc_acceleration = toppra.constraint.JointAccelerationConstraint(acceleration_factors * ACCELERATION_CONSTRAINTS)
            toppra_instance = toppra.algorithm.TOPPRA(constraint_list=[pc_velocity, pc_acceleration],
                                                    path=self.__spline_path, solver_wrapper="seidel",
                                                    gridpoints=self.__grid_points, parametrizer="ParametrizeSpline")
            meta_trajectory = MetaTrajectoryTOPPRA(toppra_instance=toppra_instance)
            joint_exceed_results: List[int] = [meta_trajectory.check_all_limits(check_flags=[False, True, True, False] ,joint_index=i) for i in range(6)]
            exceed_indices = np.where(np.array(joint_exceed_results) != -1)[0]
            if exceed_indices.size == 0:
                return acceleration_factors, meta_trajectory.duration
            else:
                acceleration_factors[exceed_indices] -= 0.01
        return np.array([0.1,0.1,0.1,0.1,0.1,0.1]), 100


    def search_kinematics_greedy(self):
        pc_velocity = toppra.constraint.JointVelocityConstraint(VELOCITY_CONSTRAINT)
        best_acceleration_factors = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        min_duration = 10 * self.__transition_trajectory_len
        for order in tqdm(five_joints_orders):
            acceleration_factors = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            for joint_index in [0] + order: # search joint one by one
                acceleration_factors, duration = self.__search_joint_greedy(joint_index, pc_velocity, acceleration_factors)
            # testify the kinematics results
            acceleration_factors, duration = self.__testify_all_joints(pc_velocity, acceleration_factors)
            if duration < min_duration:
                min_duration = duration
                best_acceleration_factors = acceleration_factors
        kinematics_params = {"joint_vel_coef": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            "joint_acc_coef": list(best_acceleration_factors),
                            "joint_jerk_coef": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            "traj_planning_time": min_duration}
        return kinematics_params
    
    def __testify_detail_trajectory(self, toppra_constraint_list: List[toppra.constraint.Constraint]) -> MetaTrajectoryTOPPRA: #only consider position limit 
        insert_time = len(self.__transition_way_points)
        way_points = np.asarray(self.__transition_way_points)
        while insert_time > 0:
            insert_time -= 1
            spline_path = toppra.SplineInterpolator(np.linspace(0, 1, len(way_points)), way_points, bc_type="natural")
            toppra_instance = toppra.algorithm.TOPPRA(constraint_list=toppra_constraint_list,
                                                path=spline_path,
                                                gridpoints=np.linspace(0, spline_path.duration, NUM_POINTS_RATIO * (len(way_points) - 1) + 1),
                                                solver_wrapper="seidel",
                                                parametrizer="ParametrizeSpline")
            meta_trajectory_instance = MetaTrajectoryTOPPRA(toppra_instance=toppra_instance)
            # check position result
            position_check = meta_trajectory_instance.check_position_limit()
            if position_check == -1:
                self.__transition_way_points = way_points
                return meta_trajectory_instance
            else:
                way_index_list = meta_trajectory_instance.get_waypoint_index_list(num_ratio=NUM_POINTS_RATIO)
                insert_index = np.where(np.asarray(way_index_list) > position_check)[0][0]
                insert_point = (way_points[insert_index - 1,:] + way_points[insert_index,:]) / 2        
                way_points = np.insert(way_points, insert_index, insert_point, axis=0)
                self.__trajectory_transition_states.insert(insert_index, -2)
        return False
    
    def generate_available_detail_trajectory(self, acc_rate:np.ndarray, speed_ratio: float = 1) -> Union[pd.DataFrame, bool]:
        pc_velocity = toppra.constraint.JointVelocityConstraint(VELOCITY_CONSTRAINT)
        pc_acceleration = toppra.constraint.JointAccelerationConstraint(ACCELERATION_CONSTRAINTS * acc_rate)
        meta_trajectory_instance = self.__testify_detail_trajectory(toppra_constraint_list=[pc_velocity, pc_acceleration])
        if meta_trajectory_instance:
            detail_trajectory_without_shot, final_position_points = meta_trajectory_instance.generate_detail_trajectory(speed_ratio=speed_ratio)
            way_index_list = meta_trajectory_instance.get_waypoint_index_list(num_ratio=NUM_POINTS_RATIO, speed_ratio=speed_ratio)
            way_index_list[-1] = len(detail_trajectory_without_shot) - 1
            for trajectory_index, io_addr_str in zip(way_index_list, self.__trajectory_transition_states):
                if isinstance(io_addr_str, str):
                    detail_trajectory_without_shot.loc[trajectory_index - 8, 'do_port'] = io_addr_str
                    detail_trajectory_without_shot.loc[trajectory_index, 'do_port'] = io_addr_str
                    detail_trajectory_without_shot.loc[trajectory_index, 'do_state'] = 1
                else:
                    detail_trajectory_without_shot.loc[trajectory_index, 'do_state'] = io_addr_str

            return detail_trajectory_without_shot
        return False

if __name__ == "__main__":
    start = time.time()
    raw_trajectory_instance = RawTrajectory(trajectory_name="test_trajectory", trajectory_path="/home/chibai/host_dir/robot_controller_tmp/trajectory/test_trajectory.json")
    #kinematics = raw_trajectory_instance.search_kinematics_greedy()
    #print(kinematics)
    df: pd.DataFrame = raw_trajectory_instance.generate_available_detail_trajectory(acc_rate=[0.85,0.99,0.99,0.69,0.45,0.99], speed_ratio=0.3)
    df.to_csv("test.csv")