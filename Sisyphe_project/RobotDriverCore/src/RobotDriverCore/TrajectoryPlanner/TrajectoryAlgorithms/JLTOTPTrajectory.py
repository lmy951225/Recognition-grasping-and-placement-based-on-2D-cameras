from typing import List, Union, Tuple
from jltotp.functions import ruckig, jltotp_p2p
from jltotp.RuckigInstance import MutilPointsRuckig
import copy
import pandas as pd
import numpy as np
import numpy.typing as npt
from RobotDriverCore.RobotDriverCoreUtils.RobotConstraints import RobotConstraints
from RobotDriverCore.TrajectoryPlanner.TrajectoryAlgorithms.TrajectoryAlgorithm import TrajectoryAlgorithm

class JLTOTPTrajectory(TrajectoryAlgorithm):
    def __init__(self, way_points_or_toppra_instance: List, trajectory_constraints: RobotConstraints) -> None:
        super().__init__(way_points_or_toppra_instance, trajectory_constraints)
        self.__velocity_boundary = np.asarray([-self._trajectory_constraints.velocity_limit, self._trajectory_constraints.velocity_limit])
        self.__acceleration_boundary = np.asarray([-self._trajectory_constraints.acceleration_limit, self._trajectory_constraints.acceleration_limit])
        self.__jerk_boundary = np.asarray([-self._trajectory_constraints.jerk_limit, self._trajectory_constraints.jerk_limit])

    '''
    def generate_whole_trajectory(self, speed_ratio: float) -> Tuple[pd.DataFrame, List[int]]:
        time_step = self._time_step
        time_points, position_points, velocity_points, acceleration_points, jerk_points, way_index_list \
              = ruckig(thetas_list=np.asarray(self._way_points), v_boundary=self.__velocity_boundary, a_boundary=self.__acceleration_boundary, j_boundary=self.__jerk_boundary, T_s=time_step, scaling_factor=speed_ratio)
        csv_trajectory = self._information2csv_trajectory(time_points, np.asarray(position_points).T, np.asarray(velocity_points).T, np.asarray(acceleration_points).T, np.asarray(jerk_points).T, 1)
        return csv_trajectory, [i for i in range(len(way_index_list)) if way_index_list[i] == 1]
    '''
    def generate_whole_trajectory(self, transition_velocity_coefficients: List[float], transition_standing_times: List[float], speed_ratio: float) -> Tuple[pd.DataFrame, List[int]]:
        ruckig_instance = MutilPointsRuckig(thetas_list=np.asarray(self._way_points), v_boundary=self.__velocity_boundary, a_boundary=self.__acceleration_boundary, j_boundary=self.__jerk_boundary, T_s=self._time_step, min_dura=0.05)
        ruckig_instance.determine_VA(np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6))
        best_v_list = copy.deepcopy(ruckig_instance.final_V)
        ruckig_instance.final_V = np.matmul(best_v_list.T, transition_velocity_coefficients).T
        T, Q, V, A, J, shot_flag = ruckig_instance.multi_points_ruckig()
        T, Q, V, A, J, shot_flag = ruckig_instance.scale_time(T, Q, V, A, J, shot_flag, speed_ratio)
        csv_trajectory = self._information2csv_trajectory(T, np.asarray(Q).T, np.asarray(V).T, np.asarray(A).T, np.asarray(J).T, 1)
        return csv_trajectory, [i for i in range(len(shot_flag)) if shot_flag[i] == 1]

        
