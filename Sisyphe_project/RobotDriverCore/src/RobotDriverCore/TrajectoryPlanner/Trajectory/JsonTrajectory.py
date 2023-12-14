from typing import List, Union, Tuple
import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
import json
import copy
import numpy as np
import numpy.typing as npt
import toppra
import pandas as pd
from tqdm import tqdm
from RobotDriverCore.TrajectoryPlanner.Trajectory.Trajectory import Trajectory
from RobotDriverCore.RobotDriverCoreUtils.RobotConstraints import RobotConstraints
from RobotDriverCore.TrajectoryPlanner.TrajectoryAlgorithms.MetaTrajectoryTOPPRA import MetaTrajectoryTOPPRA
from RobotDriverCore.TrajectoryPlanner.TrajectoryAlgorithms.RuckigTrajectoryP2P import RuckigTrajectoryP2P

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


class JsonTrajectory(Trajectory):
    def __init__(self, trajectory_name: str, trajectory_path: str) -> None:
        super().__init__(trajectory_name, trajectory_path)
        self.__way_points = self._trajectory_content["flying_shots"][trajectory_name]["traj_waypoints"]# only working for shot_flags set
        self.__transition_way_points = self._trajectory_content["flying_shots"][trajectory_name]["trajectory_transition_waypoints"]
        self.__trajectory_transition_states:List[Union[int, str]] = self._trajectory_content["flying_shots"][trajectory_name]["trajectory_transition_states"]
        self.__transition_trajectory_len = len(self.__transition_way_points)
        self.__offset_values: List[int] = self._trajectory_content["flying_shots"][trajectory_name]["offset_values"]
        self.__transition_offset_values: List[int] = copy.deepcopy(self.__offset_values)
        if 'speed_coefficients' in self._trajectory_content["flying_shots"][trajectory_name] and self._trajectory_content["flying_shots"][trajectory_name]['speed_coefficients']:
            self.__transition_velocity_coefficients: List[float] = self._trajectory_content["flying_shots"][trajectory_name]['speed_coefficients']
        else:
            self.__transition_velocity_coefficients: List[float] =  [1] * len(self.__way_points)
        if 'standing_times' in self._trajectory_content["flying_shots"][trajectory_name] and self._trajectory_content["flying_shots"][trajectory_name]['standing_times']:
            self.__transition_standing_times: List[float] = self._trajectory_content["flying_shots"][trajectory_name]['standing_times']
        else:
            self.__transition_standing_times: List[float] = [0] * len(self.__way_points)
        self.__spline_path = toppra.SplineInterpolator(np.linspace(0, 1, self.__transition_trajectory_len), self.__transition_way_points, bc_type="natural")
        self.__grid_points = np.linspace(0, self.__spline_path.duration, NUM_POINTS_RATIO * (self.__transition_trajectory_len - 1) + 1)

    @property
    def way_points(self) -> List[List[float]]:
        return self.__way_points

    @property
    def first_point(self):
        return self.__transition_way_points[0]
    
    def __search_joint_greedy(self, joint_index: int, trajectory_constraints: RobotConstraints, acceleration_factors: npt.NDArray[np.float_]) -> Tuple[np.ndarray, float]:
        acceleration_factor_max = 100
        acceleration_factor_min = 1
        pc_velocity = toppra.constraint.JointVelocityConstraint(trajectory_constraints.velocity_limit)
        while acceleration_factor_min < acceleration_factor_max:
            acceleration_factor_mid = int((acceleration_factor_max + acceleration_factor_min) / 2)
            acceleration_factors[joint_index] = acceleration_factor_mid * 0.01
            pc_acceleration = toppra.constraint.JointAccelerationConstraint(acceleration_factors * trajectory_constraints.acceleration_limit)
            toppra_instance = toppra.algorithm.TOPPRA(constraint_list=[pc_velocity, pc_acceleration],
                                                    path=self.__spline_path, solver_wrapper="seidel", 
                                                    gridpoints=self.__grid_points, parametrizer="ParametrizeSpline")
            meta_trajectory = MetaTrajectoryTOPPRA(way_points_or_toppra_instance=toppra_instance, trajectory_constraints=trajectory_constraints)
            exceed_result = meta_trajectory.check_all_limit(joint_index=joint_index, check_flags=[False, True, True, True])
            if exceed_result == -1:
                duration = meta_trajectory.duration
                acceleration_factor_min = acceleration_factor_mid + 1
            else:
                acceleration_factor_max = acceleration_factor_mid
        
        if exceed_result != -1:
            acceleration_factors[joint_index] -= 0.01
        
        return acceleration_factors, duration

    def __testify_all_joints(self, trajectory_constraints: RobotConstraints, acceleration_factors: npt.NDArray[np.float_]) -> Tuple[npt.NDArray[np.float_], float]:
        pc_velocity = toppra.constraint.JointVelocityConstraint(trajectory_constraints.velocity_limit)
        for _ in range(100):
            pc_acceleration = toppra.constraint.JointAccelerationConstraint(acceleration_factors * trajectory_constraints.acceleration_limit)
            toppra_instance = toppra.algorithm.TOPPRA(constraint_list=[pc_velocity, pc_acceleration],
                                                    path=self.__spline_path, solver_wrapper="seidel",
                                                    gridpoints=self.__grid_points, parametrizer="ParametrizeSpline")
            meta_trajectory = MetaTrajectoryTOPPRA(way_points_or_toppra_instance=toppra_instance, trajectory_constraints=trajectory_constraints)
            joint_exceed_results: List[int] = [meta_trajectory.check_all_limit(joint_index=i, check_flags=[False, True, True, True]) for i in range(6)]
            exceed_indices = np.where(np.array(joint_exceed_results) != -1)[0]
            if exceed_indices.size == 0:
                return acceleration_factors, meta_trajectory.duration
            else:
                acceleration_factors[exceed_indices[0]] -= 0.01
        return np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), 100



    def search_kinematics_params(self, trajectory_constraints: RobotConstraints):
        best_acceleration_factors = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        min_duration = 10 * self.__transition_trajectory_len
        best0factors, _ = self.__search_joint_greedy(joint_index=0, trajectory_constraints=trajectory_constraints, acceleration_factors=[1.0, 1.0 ,1.0 ,1.0 ,1.0 ,1.0])
        for order in tqdm(five_joints_orders):
            acceleration_factors = copy.deepcopy(best0factors) # copy again
            for joint_index in order: # search joint one by one
                acceleration_factors, duration = self.__search_joint_greedy(joint_index, trajectory_constraints, acceleration_factors)
            # testfy the kinematics results
            acceleration_factors, duration = self.__testify_all_joints(trajectory_constraints, acceleration_factors)
            if duration < min_duration:
                min_duration = duration
                best_acceleration_factors = acceleration_factors
        # generate kinematics parameters
        kinematics_params = {"joint_vel_coef": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            "joint_acc_coef": list(best_acceleration_factors),
                            "joint_jerk_coef": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            "traj_planning_time": min_duration}
        return kinematics_params
    
    def __testify_csv_trajectory(self, toppra_constraint_list: List[toppra.constraint.Constraint], trajectory_constraints: RobotConstraints) -> MetaTrajectoryTOPPRA: #only consider position limit 
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
            meta_trajectory_instance = MetaTrajectoryTOPPRA(way_points_or_toppra_instance=toppra_instance, trajectory_constraints=trajectory_constraints)
            # check position result
            position_check = meta_trajectory_instance.check_position_limit()
            if position_check == -1:
                self.__transition_way_points = way_points
                self.__transition_trajectory_len = len(self.__transition_way_points)
                return meta_trajectory_instance
            else:
                way_index_list = meta_trajectory_instance.get_waypoint_index_list(num_ratio=NUM_POINTS_RATIO)
                insert_index = np.where(np.asarray(way_index_list) > position_check)[0][0]
                insert_point = (way_points[insert_index - 1,:] + way_points[insert_index,:]) / 2        
                way_points = np.insert(way_points, insert_index, insert_point, axis=0)
                self.__trajectory_transition_states.insert(insert_index, -2)
        return int(insert_index)

    def __generate_correponding_toppra_csv(self, acc_rate: np.ndarray, trajectory_constraints: RobotConstraints, speed_ratio: float = 1) -> Union[int, Tuple[pd.DataFrame, List[int]]]:
        pc_velocity = toppra.constraint.JointVelocityConstraint(trajectory_constraints.velocity_limit)
        pc_acceleration = toppra.constraint.JointAccelerationConstraint(trajectory_constraints.acceleration_limit * acc_rate)
        meta_trajectory_instance = self.__testify_csv_trajectory(toppra_constraint_list=[pc_velocity, pc_acceleration], trajectory_constraints=trajectory_constraints)
        if isinstance(meta_trajectory_instance, MetaTrajectoryTOPPRA):
            csv_trajectory_without_shot = meta_trajectory_instance.generate_detail_trajectory(speed_ratio=speed_ratio)
            way_index_list = meta_trajectory_instance.get_waypoint_index_list(num_ratio=NUM_POINTS_RATIO, speed_ratio=speed_ratio)
            way_index_list[-1] = len(csv_trajectory_without_shot) - 1
            return csv_trajectory_without_shot, way_index_list
        else:
            return len([p for p in self.__trajectory_transition_states[:meta_trajectory_instance] if p != -2])
    
    def __add_io_port_state(self, csv_trajectory_without_shot: pd.DataFrame, way_index_list: List[int], off_signal_translation: int, way_states: List[Union[str, int]]) -> pd.DataFrame:
        for trajectory_index, io_addr_str, io_offset in zip(way_index_list, way_states, self.__transition_offset_values):
            if isinstance(io_addr_str, str):
                # get normal on and off signal index
                on_signal_index = trajectory_index + io_offset
                off_signal_index = on_signal_index + off_signal_translation
                # check last point
                csv_rows = len(csv_trajectory_without_shot)
                if off_signal_index >= csv_rows - 2:
                    off_signal_index = csv_rows - 1
                    on_signal_index = off_signal_index - off_signal_translation
                # set real off singal and on signal
                csv_trajectory_without_shot.loc[off_signal_index, 'do_port'] = io_addr_str
                csv_trajectory_without_shot.loc[on_signal_index, 'do_port'] = io_addr_str
                csv_trajectory_without_shot.loc[on_signal_index, 'do_state'] = 1
            else:
                csv_trajectory_without_shot.loc[trajectory_index, 'do_state'] = io_addr_str
        return csv_trajectory_without_shot

    def generate_available_csv_trajectory(self, acc_rate: np.ndarray, trajectory_constraints: RobotConstraints, speed_ratio: float = 1, trajectory_approach: int = 1) -> Union[pd.DataFrame, int]:
        off_signal_translation = round(0.016 / trajectory_constraints.time_step)
        # trajectory_approach (1 -> pure toppra| 2 -> toppra + ruckig| 3 -> toppra + JLTOTP| 4 -> pure JLTOTP)
        ################################################4. pure JLTOTP#######################################
        if trajectory_approach == 4:
            from RobotDriverCore.TrajectoryPlanner.TrajectoryAlgorithms.JLTOTPTrajectory import JLTOTPTrajectory
            jltotp = JLTOTPTrajectory(self.__way_points, trajectory_constraints=trajectory_constraints)
            jltotp_csv, jltotp_way_index = jltotp.generate_whole_trajectory(transition_velocity_coefficients=self.__transition_velocity_coefficients, transition_standing_times=self.__transition_standing_times, speed_ratio=speed_ratio)
            way_states = [s for s in self.__trajectory_transition_states if s != -2]
            return self.__add_io_port_state(csv_trajectory_without_shot=jltotp_csv, way_index_list=jltotp_way_index, off_signal_translation=off_signal_translation, way_states=way_states)
        ################################################################################################
        ##############################################1. pure toppra####################################
        toppra_speed = speed_ratio if trajectory_approach == 1 else 1
        results = self.__generate_correponding_toppra_csv(acc_rate, trajectory_constraints, toppra_speed)
        if isinstance(results, int):
            return results
        else:
            csv_trajectory_without_shot, way_index_list = results
        # add transition points(state: -2) parameters
        for i in range(self.__transition_trajectory_len):
            if self.__trajectory_transition_states[i] == -2:
                self.__transition_offset_values.insert(i, 0)
                self.__transition_velocity_coefficients.insert(i, 1)
                self.__transition_standing_times.insert(i, 0)
        if trajectory_approach == 1: # approach 1: pure toppra, can not start or end with 0 acceleration
            return self.__add_io_port_state(csv_trajectory_without_shot, way_index_list, off_signal_translation, way_states=self.__trajectory_transition_states)
        ##########################################################################################################
        if trajectory_approach == 2:
            ruckig_p2p = RuckigTrajectoryP2P(way_points_or_toppra_instance=self.__transition_way_points, trajectory_constraints=trajectory_constraints)
            toppra_velocity_points = [np.zeros(6)] + [csv_trajectory_without_shot.iloc[index].iloc[7:13].to_numpy() for index in way_index_list[1:-1]] + [np.zeros(6)]
            toppra_acceleration_points = [np.zeros(6)] + [csv_trajectory_without_shot.iloc[index].iloc[13:19].to_numpy() for index in way_index_list[1:-1]] + [np.zeros(6)]
            best_velocity_points, best_acceleration_points = ruckig_p2p.search_best_parameters(target_points_velocity=toppra_velocity_points, target_points_acceleration=toppra_acceleration_points)
            for i in range(1, self.__transition_trajectory_len - 1):
                best_velocity_points[i] = best_velocity_points[i] * self.__transition_velocity_coefficients[i]
                if self.__transition_velocity_coefficients[i] < 0.01 and self.__transition_standing_times[i] > 0:
                    best_acceleration_points[i] = np.zeros(6)
            ruckig_csv_trajectory, ruckig_way_index = ruckig_p2p.generate_whole_trajectory(target_points_velocity=best_velocity_points, target_points_acceleration=best_acceleration_points, target_standing_times=self.__transition_standing_times, speed_ratio=speed_ratio)
            return self.__add_io_port_state(ruckig_csv_trajectory, ruckig_way_index, off_signal_translation, self.__trajectory_transition_states)
            
        ##########################################################################################################
        raise ValueError("not support trajectory approach {}, please check!!!!!!!!!!!!!".format(trajectory_approach))
        

if __name__ == '__main__':
    from RobotDriverCore.RobotDriverCoreUtils.RobotConstraints import RobotConstraints
    robot_constraint = RobotConstraints() # unit is radian
    robot_constraint.time_step = 0.001
    robot_constraint.max_position       =np.asarray([ 2.94,  1.72,  3.47,  3.29,  1.98,  6.26])
    robot_constraint.min_position       =np.asarray([-2.94, -1.72, -3.47, -3.29, -1.98, -6.26])
    robot_constraint.velocity_limit     =np.asarray([5.71, 4.56, 5.71, 7.75, 6.97, 10.46])
    robot_constraint.acceleration_limit =np.asarray([23,      19,   23,     31,    27,    39])
    robot_constraint.jerk_limit         =np.asarray([23,      19,   23,     31,    27,    39]) * 8
    json_trajectory = JsonTrajectory(trajectory_name='4-1-1', trajectory_path='4-1-/4-1-1.json')
    acc_rate = json_trajectory.search_kinematics_params(robot_constraint)['joint_acc_coef']
    #acc_rate = [0.29, 0.26, 0.21, 0.99, 0.99, 0.99]
    print(acc_rate)
    csv  = json_trajectory.generate_available_csv_trajectory(acc_rate=acc_rate, trajectory_constraints=robot_constraint, speed_ratio=0.7, trajectory_approach=2) 
    csv.to_csv("4-1-/4-1-1_approach2half.csv", index=False)
    #csv  = json_trajectory.generate_available_csv_trajectory(acc_rate=acc_rate, trajectory_constraints=robot_constraint, speed_ratio=1, trajectory_approach=3) 
    #csv.to_csv("9-1-5_approach3_100%.csv", index=False)
    #csv  = json_trajectory.generate_available_csv_trajectory(acc_rate=acc_rate, trajectory_constraints=robot_constraint, speed_ratio=1, trajectory_approach=4) 
    #csv.to_csv("9-1-5_approach4.csv", index=False)