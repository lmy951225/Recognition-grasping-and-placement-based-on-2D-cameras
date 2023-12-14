from typing import Union, List, Tuple
import math
import toppra
toppra.setup_logging("WARN")
from toppra.interpolator import AbstractGeometricPath
import numpy as np
import numpy.typing as npt
import pandas as pd
from RobotDriverCore.RobotDriverCoreUtils.RobotConstraints import RobotConstraints
from RobotDriverCore.TrajectoryPlanner.TrajectoryAlgorithms.TrajectoryAlgorithm import TrajectoryAlgorithm

class MetaTrajectoryTOPPRA(TrajectoryAlgorithm):
    def __init__(self, way_points_or_toppra_instance: Union[toppra.algorithm.TOPPRA, List], trajectory_constraints: RobotConstraints) -> None:
        super().__init__(way_points_or_toppra_instance, trajectory_constraints)
        if type(way_points_or_toppra_instance) == list:
            self.__position_points = np.asarray(way_points_or_toppra_instance)
        else:
            self.__jnt_traj = way_points_or_toppra_instance.compute_trajectory()
            self.__duration = self.__jnt_traj.duration
            time_points = np.linspace(0, self.__duration, int(self.__duration // self._time_step) + 1)
            self.__position_points: npt.NDArray[np.float_] = self.__jnt_traj(time_points)
            self.__velocity_points: npt.NDArray[np.float_] = self.__jnt_traj(time_points, 1)
            self.__acceleration_points: npt.NDArray[np.float_] = self.__jnt_traj(time_points, 2)
            self.__jerk_points: npt.NDArray[np.float_] = np.gradient(self.__acceleration_points, self._time_step, axis=0)

    @property
    def toppra_joint_trajectory(self) -> AbstractGeometricPath:
        return self.__jnt_traj

    @property
    def duration(self) -> float:
        return self.__duration

    @property
    def position_points(self) -> npt.NDArray[np.float_]:
        return self.__position_points
    
    @property
    def velocity_points(self) -> npt.NDArray[np.float_]:
        return self.__velocity_points
    
    @property
    def acceleration_points(self) -> npt.NDArray[np.float_]:
        return self.__acceleration_points
    
    @property
    def jerk_points(self) -> npt.NDArray[np.float_]:
        return self.__jerk_points
        
    def get_waypoint_index_list(self, num_ratio: int, speed_ratio: float = 1) -> List[int]:
        results: List[int] = [0]
        grid_points_len = len(self.__jnt_traj.waypoints[0])
        for i in range(num_ratio, grid_points_len + 2, num_ratio):
            time_point = self.__jnt_traj.waypoints[0][i]
            results.append(round(time_point / self._time_step / speed_ratio))
        return results
    
    @staticmethod
    def __check_value_exceeded(joints_values: npt.NDArray[np.float_], min_limit: npt.NDArray[np.float_], max_limit: npt.NDArray[np.float_], joint_index: int = -1) -> int:
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
    
    @staticmethod
    def __zero_first_last_row(matrix: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        matrix[0, :] = 0
        matrix[-1,:] = 0
        return matrix
        
    def check_position_limit(self, joint_index:int = -1, threshold: float = 0) -> int:
        return self.__check_value_exceeded(joints_values=self.__position_points, min_limit=self._trajectory_constraints.min_position-threshold, max_limit=self._trajectory_constraints.max_position+threshold, joint_index=joint_index)
    
    def check_velocity_limit(self, joint_index:int = -1, threshold: float = 0.001) -> int:
        return self.__check_value_exceeded(joints_values=self.__velocity_points, min_limit=-self._trajectory_constraints.velocity_limit-threshold, max_limit=self._trajectory_constraints.velocity_limit+threshold, joint_index=joint_index)
    
    def check_acceleration_limit(self, joint_index:int = -1, threshold: float = 0.01) -> int:
        return self.__check_value_exceeded(joints_values=self.__acceleration_points, min_limit=-self._trajectory_constraints.acceleration_limit-threshold, max_limit=self._trajectory_constraints.acceleration_limit+threshold, joint_index=joint_index)
    
    def check_jerk_limit(self, joint_index:int = -1, threshold: float = 0.1) -> int:
        return self.__check_value_exceeded(joints_values=self.__jerk_points, min_limit=-self._trajectory_constraints.jerk_limit-threshold, max_limit=self._trajectory_constraints.jerk_limit+threshold, joint_index=joint_index)
    
    def check_all_limit(self, thresholds: List[float] = [0, 0.001, 0.01, 0.1], joint_index: int = -1, check_flags: List[bool] = [True, True, True, True]):
        results: List[int] = list()
        check_limit_functions = [self.check_position_limit, self.check_velocity_limit, self.check_acceleration_limit, self.check_jerk_limit]
        for i in range(4):
            if check_flags[i]:
                check_result = check_limit_functions[i](joint_index=joint_index, threshold=thresholds[i])
                if check_result != -1:
                    results.append(check_result)
        
        return min(results) if results else -1
        
    def generate_transition_points(self, point_states: List[Union[str, int]]) -> Union[Tuple[List[List[float]], List[int]], int]:
        discrete_number = 20000
        target_ss = np.linspace(0, 1, discrete_number)
        way_points = self.__position_points
        insert_time = len(way_points)
        while insert_time > 0:
            insert_time -= 1
            ss = np.linspace(0, 1, len(way_points))
            spline_path = toppra.SplineInterpolator(ss, way_points, bc_type="natural")
            target_values = spline_path(target_ss)
            position_check = self.__check_value_exceeded(joints_values=target_values, min_limit=self._trajectory_constraints.min_position, max_limit=self._trajectory_constraints.max_position)
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
            return len([i for i in point_states[:insert_index] if i != -2])
        
    def generate_detail_trajectory(self, speed_ratio: float = 1.0) -> pd.DataFrame:
        """generate detail trajectory without IO information

        Args:
            speed_ratio (float, optional): speed ratio for the trajectory. Defaults to 1.0.

        Returns:
           pd.DataFrame: dataframe of the trajectory
        """
        time_points = np.linspace(0, self.__duration, int(self.__duration / speed_ratio / self._time_step + 1))
        position_points: np.ndarray = self.__jnt_traj(time_points)
        velocity_points: np.ndarray = self.__jnt_traj(time_points, 1)
        acceleration_points: np.ndarray = self.__jnt_traj(time_points, 2)
        jerk_points: np.ndarray = np.gradient(acceleration_points, self._time_step, axis=0)
        return self._information2csv_trajectory(time_points, position_points, velocity_points, acceleration_points, jerk_points, speed_ratio)