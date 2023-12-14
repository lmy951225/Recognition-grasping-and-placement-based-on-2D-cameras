from typing import List, Union, Tuple
import toppra
import pandas as pd
import numpy as np
import numpy.typing as npt
from RobotDriverCore.RobotDriverCoreUtils.RobotConstraints import RobotConstraints

class TrajectoryAlgorithm():
    def __init__(self, way_points_or_toppra_instance: Union[toppra.algorithm.TOPPRA, List], trajectory_constraints: RobotConstraints) -> None:
        self._way_points = way_points_or_toppra_instance
        self._trajectory_constraints = trajectory_constraints
        self._time_step = trajectory_constraints.time_step

    @property
    def way_points(self) -> Union[toppra.algorithm.TOPPRA, List]:
        return self._way_points
    
    @property
    def trajectory_constraints(self) -> RobotConstraints:
        return self._trajectory_constraints
    
    @property
    def time_step(self) -> float:
        return self._time_step
    
    def _accumulate_standing_time(self, standing_time: float, points_index: int, time_points: List[float], position_points: List[npt.NDArray[np.float_]], velocity_points: List[npt.NDArray[np.float_]], acceleration_points: List[npt.NDArray[np.float_]], jerk_points: List[npt.NDArray[np.float_]] = None) -> int:
        current_position = position_points[-1]
        if standing_time > 0.001:
            total_waiting_steps = int(standing_time / self._time_step + 1)
            for _ in range(total_waiting_steps):
                points_index += 1
                time_points.append(points_index * self._time_step)
            position_points.extend([current_position] * total_waiting_steps)
            velocity_points.extend([np.zeros(6)] * total_waiting_steps)
            acceleration_points.extend([np.zeros(6)] * total_waiting_steps)
            if jerk_points:
                jerk_points.extend([np.zeros(6)] * total_waiting_steps)
            
        return points_index

    def _information2csv_trajectory(self, time_points: npt.NDArray[np.float_], position_points: npt.NDArray[np.float_], velocity_points: npt.NDArray[np.float_], acceleration_points: npt.NDArray[np.float_], jerk_points: npt.NDArray[np.float_], speed_ratio: float) -> pd.DataFrame:
        df_ts = pd.DataFrame(time_points, columns=['ts']) / speed_ratio
        df_p = pd.DataFrame(position_points, columns=['pts_J' + str(i) for i in range(1, 7)])
        df_v = pd.DataFrame(velocity_points, columns=['vel_J' + str(i) for i in range(1, 7)]) * speed_ratio
        df_a = pd.DataFrame(acceleration_points, columns=['acc_J' + str(i) for i in range(1, 7)])  * pow(speed_ratio, 2)
        df_j = pd.DataFrame(jerk_points, columns=['jerk_J' + str(i) for i in range(1, 7)])  * pow(speed_ratio, 3)
        detailed_trajecory = pd.concat([df_ts, df_p, df_v, df_a, df_j], axis=1)  
        detailed_trajecory.loc[:, "do_port"] = -1
        detailed_trajecory.loc[:, "do_state"] = 0
        detailed_trajecory['do_port'] = detailed_trajecory['do_port'].astype('string')
        return detailed_trajecory