from typing import List, Union, Tuple
import math
import numpy as np
import pandas as pd
import toppra
from toppra.interpolator import AbstractGeometricPath
from utils.parameter_util import TIME_STEP, POSITION_CONSTRAINT, VELOCITY_CONSTRAINT, ACCELERATION_CONSTRAINTS, JERK_CONSTRAINTS


MIN_SHOT_INTERVAL = 32

class MetaTrajectoryTOPPRA():
    def __init__(self, toppra_instance: toppra.algorithm.TOPPRA) -> None:
        if type(toppra_instance) == list:
            self.__position_points = np.asarray(toppra_instance)
        else:
            self.__jnt_traj = toppra_instance.compute_trajectory()
            self.__duration = self.__jnt_traj.duration
            time_points = np.linspace(0, self.__duration, int(self.__duration // TIME_STEP) + 1)
            self.__position_points: np.ndarray = self.__jnt_traj(time_points)
            self.__velocity_points: np.ndarray = self.__jnt_traj(time_points, 1)
            self.__acceleration_points: np.ndarray = self.__jnt_traj(time_points, 2)
            self.__jerk_points: np.ndarray = np.gradient(self.__acceleration_points, TIME_STEP, axis=0)
            
    @property
    def toppra_joint_trajectory(self) -> AbstractGeometricPath:
        return self.__jnt_traj

    @property
    def duration(self) -> float:
        return self.__duration

    @property
    def position_points(self) -> np.ndarray:
        return self.__position_points
    
    @property
    def velocity_points(self) -> np.ndarray:
        return self.__velocity_points
    
    @property
    def acceleration_points(self) -> np.ndarray:
        return self.__acceleration_points
    
    @property
    def jerk_points(self) -> np.ndarray:
        return self.__jerk_points
    
    def get_waypoint_index_list(self, num_ratio: int, speed_ratio: float = 1) -> List[int]:
        results: List[int] = [0]
        grid_points_len = len(self.__jnt_traj.waypoints[0])
        for i in range(num_ratio, grid_points_len + 2, num_ratio):
            time_point = self.__jnt_traj.waypoints[0][i]
            results.append(round(time_point / TIME_STEP / speed_ratio))
        return results

    @staticmethod
    def __check_value_exceeded(joints_values: np.ndarray, min_limit: np.ndarray, max_limit: np.ndarray, joint_index: int = -1) -> int:
        if 0 <= joint_index < 6:
            joints_values = joints_values[:, joint_index]
            min_limit = min_limit[joint_index]
            max_limit = max_limit[joint_index]
        available_values = np.logical_and(joints_values > min_limit, joints_values < max_limit)
        if len(available_values.shape) == 1:
            joints_flags = available_values
        else:
            joints_flags = available_values.all(axis = 1)
        exceed_indices = np.where(joints_flags==False)[0]
        if exceed_indices.size == 0:
            return -1
        else:
            return int(exceed_indices[0])
    
    def check_position_limit(self, joint_index:int = -1, threshold: float = 0) -> int:
        return self.__check_value_exceeded(joints_values=self.__position_points, min_limit=POSITION_CONSTRAINT[0]-threshold, max_limit=POSITION_CONSTRAINT[1]+threshold, joint_index=joint_index)
    
    def check_velocity_limit(self, joint_index:int = -1, threshold: float = 0.001) -> int:
        return self.__check_value_exceeded(joints_values=self.__velocity_points, min_limit=-VELOCITY_CONSTRAINT-threshold, max_limit=VELOCITY_CONSTRAINT+threshold, joint_index=joint_index)
    
    def check_acceleration_limit(self, joint_index:int = -1, threshold: float = 0.01) -> int:
        return self.__check_value_exceeded(joints_values=self.__acceleration_points, min_limit=-ACCELERATION_CONSTRAINTS-threshold, max_limit=ACCELERATION_CONSTRAINTS+threshold, joint_index=joint_index)
    
    def check_jerk_limit(self, joint_index:int = -1, threshold: float = 0.1) -> int:
        return self.__check_value_exceeded(joints_values=self.__jerk_points, min_limit=-JERK_CONSTRAINTS-threshold, max_limit=JERK_CONSTRAINTS+threshold, joint_index=joint_index)
    
    def check_all_limits(self, check_flags: List[bool] = [True, True, True, True], thresholds: List[float] = [0, 0.001, 0.01, 0.1], joint_index: int = -1) -> int:
        results: List[int] = list()
        check_limit_functions = [self.check_position_limit, self.check_velocity_limit, self.check_acceleration_limit, self.check_jerk_limit]
        for i in range(4):
            if check_flags[i]:
                check_result = check_limit_functions[i](joint_index=joint_index, threshold=thresholds[i])
                if check_result != -1:
                    results.append(check_result)
        if results:
            return min(results)
        else:
            return -1
        

    def generate_detail_trajectory(self, speed_ratio: float = 1) ->  Tuple[pd.DataFrame, np.ndarray]:
        time_points = np.linspace(0, self.__duration, int((self.__duration // TIME_STEP + 1) / speed_ratio))
        position_points: np.ndarray = self.__jnt_traj(time_points)
        if abs(speed_ratio - 1) < 0.01: # no slowdonw
            velocity_points: np.ndarray = self.__jnt_traj(time_points, 1)
            acceleration_points: np.ndarray = self.__jnt_traj(time_points, 2)
        else:
            velocity_points: np.ndarray = np.gradient(position_points, TIME_STEP, axis=0)
            acceleration_points: np.ndarray = np.gradient(velocity_points, TIME_STEP, axis=0)
        jerk_points: np.ndarray = np.gradient(acceleration_points, TIME_STEP, axis=0)
        real_time_points = np.linspace(0, len(time_points) * TIME_STEP ,len(time_points))          
        jt_cols = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        df_ts = pd.DataFrame(real_time_points, columns=['ts'])
        df_pts = pd.DataFrame(position_points, columns=['pts_'+c for c in jt_cols])
        df_vel = pd.DataFrame(velocity_points, columns=['vel_'+c for c in jt_cols])
        df_acc = pd.DataFrame(acceleration_points, columns=['acc_'+c for c in jt_cols])
        df_jerk = pd.DataFrame(jerk_points, columns=['jerk_'+c for c in jt_cols])
        detailed_trajecory = pd.concat([df_ts, df_pts, df_vel, df_acc, df_jerk], axis=1)  
        detailed_trajecory.loc[:, "do_port"] = -1
        detailed_trajecory.loc[:, "do_state"] = 0
        return detailed_trajecory, position_points
    
    def generate_transition_points(self, point_states: List[Union[str, int]]) -> Union[Tuple[List[List[float]], List[int]], bool]:
        discrete_number = 20000
        target_ss = np.linspace(0, 1, discrete_number)
        way_points = self.__position_points
        insert_time = len(way_points)
        while insert_time > 0:
            insert_time -= 1
            ss = np.linspace(0, 1, len(way_points))
            spline_path = toppra.SplineInterpolator(ss, way_points, bc_type="natural")
            target_values = spline_path(target_ss)
            position_check = self.__check_value_exceeded(joints_values=target_values, min_limit=POSITION_CONSTRAINT[0], max_limit=POSITION_CONSTRAINT[1])
            if position_check == -1:
                break
            else:
                insert_index = math.floor(len(way_points) * position_check / discrete_number)
                insert_point = (way_points[insert_index - 1,:] + way_points[insert_index,:]) / 2        
                way_points = np.insert(way_points, insert_index, insert_point, axis=0)
                point_states.insert(insert_index, -2)
        # return final results
        if position_check == -1:
            return [list(w) for w in way_points], point_states
        else:
            return False