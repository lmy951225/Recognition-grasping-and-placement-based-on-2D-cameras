from typing import List, Tuple, Union
import copy
import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from ruckig import InputParameter, OutputParameter, Result, Ruckig, Trajectory
from RobotDriverCore.TrajectoryPlanner.TrajectoryAlgorithms.TrajectoryAlgorithm import TrajectoryAlgorithm
from RobotDriverCore.RobotDriverCoreUtils.RobotConstraints import RobotConstraints

class SeachVAengine():
    def __init__(self, position_points: Union[List[np.ndarray], np.ndarray], velocity_points: List[np.ndarray], acceleration_points: List[np.ndarray], trajectory_constraints: RobotConstraints) -> None:
        self.__p0, self.__p1, self.__p2 = position_points
        self.__v0, self.__v1, self.__v2 = velocity_points
        self.__a0, self.__a1, self.__a2 = acceleration_points
        self._trajectory_constraints = trajectory_constraints
        self.__search_range_ratio = 0.08

    @property
    def v1(self):
        return self.__v1
    
    @property
    def a1(self):
        return self.__a1

    def __offline_ruckig_p2p(self, p1, v1, a1, p2, v2, a2) -> Union[None, Trajectory]:
        inp = InputParameter(6)
        # current
        inp.current_position = p1
        inp.current_velocity = v1
        inp.current_acceleration = a1
        # target
        inp.target_position = p2
        inp.target_velocity = v2
        inp.target_acceleration = a2
        # constraints
        inp.min_position = self._trajectory_constraints.min_position.tolist()
        inp.max_position = self._trajectory_constraints.max_position.tolist()
        inp.max_velocity = self._trajectory_constraints.velocity_limit.tolist()
        inp.max_acceleration = self._trajectory_constraints.acceleration_limit.tolist()
        inp.max_jerk = self._trajectory_constraints.jerk_limit.tolist()
        otg = Ruckig(6)
        trajectory = Trajectory(6)
        result = otg.calculate(inp, trajectory)
        if result != Result.Working:
            return None
        else:
            return trajectory
    
    def __calculate_whole_time(self, v1, a1) -> float:
        trajectory0 = self.__offline_ruckig_p2p(p1=self.__p0, v1=self.__v0, a1=self.__a0, p2=self.__p1, v2=v1, a2=a1)
        trajectory1 = self.__offline_ruckig_p2p(p1=self.__p1, v1=v1, a1=a1, p2=self.__p2, v2=self.__v2, a2=self.__a2)
        if trajectory0 is None or trajectory1 is None:
            return 10000
        else:
            return trajectory0.duration + trajectory1.duration

    def __do_search(self, search_index: int, target_time: float, is_velocity: bool = True) -> Tuple[np.ndarray, float]:
        low, high = 1, 100
        if is_velocity:
            search_range = self.__v1[search_index] * self.__search_range_ratio
        else:
            search_range = self.__a1[search_index] * self.__search_range_ratio
        while low < high:
            v1, a1 = copy.deepcopy(self.__v1), copy.deepcopy(self.__a1)
            mid = (high + low) // 2
            search_diff = mid * search_range / 100
            if is_velocity:
                v1[search_index] = v1[search_index] + search_diff
            else:
                a1[search_index] = a1[search_index] + search_diff
            if (abs(v1[search_index]) <= self._trajectory_constraints.velocity_limit[search_index]) and (abs(a1[search_index]) <= self._trajectory_constraints.acceleration_limit[search_index]):
                total_time = self.__calculate_whole_time(v1, a1)
                if total_time < target_time:
                    return (v1, total_time) if is_velocity else (a1, total_time)
            high = mid
        return (self.__v1, target_time) if is_velocity else (self.__a1, target_time)


    def search_betterVA(self) -> Tuple[np.ndarray, np.ndarray]:
        trajectory0 = self.__offline_ruckig_p2p(p1=self.__p0, v1=self.__v0, a1=self.__a0, p2=self.__p1, v2=self.__v1, a2=self.__a1)
        trajectory1 = self.__offline_ruckig_p2p(p1=self.__p1, v1=self.__v1, a1=self.__a1, p2=self.__p2, v2=self.__v2, a2=self.__a2)
        # find the longest duration for 
        search_index0 = trajectory0.independent_min_durations.index(max(trajectory0.independent_min_durations))
        search_index1 = trajectory1.independent_min_durations.index(max(trajectory1.independent_min_durations))
        current_best_time = trajectory0.duration + trajectory1.duration
        # do first v1
        v1, current_best_time = self.__do_search(search_index=search_index0, target_time=current_best_time, is_velocity=True)
        self.__v1 = v1
        # do first a1
        a1, current_best_time = self.__do_search(search_index=search_index0, target_time=current_best_time, is_velocity=False)
        self.__a1 = a1
        # do second v1
        v1, current_best_time = self.__do_search(search_index=search_index1, target_time=current_best_time, is_velocity=True)
        self.__v1 = v1
        # do second a2
        a1, current_best_time = self.__do_search(search_index=search_index1, target_time=current_best_time, is_velocity=False)
        self.__a1 = a1


class RuckigTrajectoryP2P(TrajectoryAlgorithm):
    def __init__(self, way_points_or_toppra_instance, trajectory_constraints: RobotConstraints) -> None:
        super().__init__(way_points_or_toppra_instance, trajectory_constraints)

    def __ruckig_p2p(self, p1, velocity1, acceleration1, p2, velocity2, acceleration2, time_step: float) -> List[OutputParameter]:
        otg = Ruckig(6, time_step)
        inp = InputParameter(6)
        out = OutputParameter(6)
        # current
        inp.current_position = p1
        inp.current_velocity = velocity1
        inp.current_acceleration = acceleration1
        # target
        inp.target_position = p2
        inp.target_velocity = velocity2
        inp.target_acceleration = acceleration2
        # constraints
        inp.min_position = self._trajectory_constraints.min_position.tolist()
        inp.max_position = self._trajectory_constraints.max_position.tolist()
        inp.max_velocity = self._trajectory_constraints.velocity_limit.tolist()
        inp.max_acceleration = self._trajectory_constraints.acceleration_limit.tolist()
        inp.max_jerk = self._trajectory_constraints.jerk_limit.tolist()
        # Generate the trajectory within the control loop
        first_output, out_list = None, []
        res = Result.Working
        while res == Result.Working:
            res = otg.update(inp, out)
            out_list.append(copy.copy(out))
            out.pass_to_input(inp)
            if not first_output:
                first_output = copy.copy(out)
        trajectory_duration = first_output.trajectory.duration
        if abs(out_list[-1].time - trajectory_duration) > abs(out_list[-2].time - trajectory_duration):
            out_list.pop() # remove last element if it's further than last second element
        return out_list

    def search_best_parameters(self, target_points_velocity:List[np.ndarray], target_points_acceleration:List[np.ndarray], search_iterations: int = 200):
        for _ in tqdm(range(search_iterations)):
            for i in range(1, len(self._way_points) - 1):
                searcher = SeachVAengine(position_points=self._way_points[i-1:i+2], velocity_points=target_points_velocity[i-1:i+2], acceleration_points=target_points_acceleration[i-1:i+2], trajectory_constraints=self._trajectory_constraints)
                searcher.search_betterVA()
                target_points_velocity[i] = searcher.v1
                target_points_acceleration[i] = searcher.a1
        return target_points_velocity, target_points_acceleration

    def workaround_ruckig_p2p(self, target_points_velocity:List[npt.NDArray[np.float_]], target_points_acceleration:List[npt.NDArray[np.float_]], target_standing_times: List[float], speed_ratio: float = 1) -> Tuple[pd.DataFrame, List[int]]:
        points_index = 0
        way_index_list: List[int] = [0]
        time_step = self._time_step * speed_ratio
        time_points: List[float] = [0]
        position_points: List[npt.NDArray[np.float_]] = [np.asarray(self._way_points[0])]
        velocity_points: List[npt.NDArray[np.float_]] = [np.asarray(target_points_velocity[0])]
        acceleration_points: List[npt.NDArray[np.float_]] = [np.asarray(target_points_acceleration[0])]
        points_index = self._accumulate_standing_time(target_standing_times[0], points_index, time_points, position_points, velocity_points, acceleration_points)
        for target_p, target_velocity, target_acceleration, standing_time in zip(self._way_points[1:], target_points_velocity[1:], target_points_acceleration[1:], target_standing_times[1:]):
            output_list = self.__ruckig_p2p(p1=position_points[-1], velocity1=velocity_points[-1], acceleration1=acceleration_points[-1],
                                            p2=list(target_p), velocity2=target_velocity.tolist(), acceleration2=target_acceleration.tolist(), time_step=time_step)
            for p in output_list:
                points_index += 1
                time_points.append(points_index * time_step)
                position_points.append(p.new_position)
                velocity_points.append(p.new_velocity)
                acceleration_points.append(p.new_acceleration)
            way_index_list.append(points_index)
            points_index = self._accumulate_standing_time(standing_time, points_index, time_points, position_points, velocity_points, acceleration_points)
        # assign last point to make end of trajectory precise
        points_index += 1
        time_points.append(points_index * time_step)
        position_points.append(np.asarray(self._way_points[-1]))
        velocity_points.append((position_points[-1] - position_points[-2]) / self._time_step)
        acceleration_points.append((velocity_points[-1] - velocity_points[-2]) / self._time_step)
        way_index_list[-1] += 1
        # estimate jerk points
        jerk_points = np.gradient(acceleration_points, self._time_step, axis=0)
        csv_trajecoty = self._information2csv_trajectory(time_points, position_points, velocity_points, acceleration_points, jerk_points, speed_ratio)
        return csv_trajecoty, way_index_list
    

    def generate_whole_trajectory(self, target_points_velocity:List[npt.NDArray[np.float_]], target_points_acceleration:List[npt.NDArray[np.float_]], target_standing_times: List[float], speed_ratio: float = 1) -> Tuple[pd.DataFrame, List[int]]:
        mini_ts = self._time_step / 100
        points_index = 0
        way_index_list: List[int] = [0]
        time_points: List[float] = [0]
        position_points: List[npt.NDArray[np.float_]] = [np.asarray(self._way_points[0])]
        velocity_points: List[npt.NDArray[np.float_]] = [np.asarray(target_points_velocity[0])]
        acceleration_points: List[npt.NDArray[np.float_]] = [np.asarray(target_points_acceleration[0])]
        points_index = self._accumulate_standing_time(0, points_index, time_points, position_points, velocity_points, acceleration_points)
        for target_p, target_velocity, target_acceleration, standing_time in zip(self._way_points[1:], target_points_velocity[1:], target_points_acceleration[1:], target_standing_times[1:]):
            output_list = self.__ruckig_p2p(p1=position_points[-1], velocity1=velocity_points[-1], acceleration1=acceleration_points[-1],
                                            p2=list(target_p), velocity2=target_velocity.tolist(), acceleration2=target_acceleration.tolist(), time_step=mini_ts)
            for p in output_list:
                points_index += 1
                time_points.append(points_index * mini_ts)
                position_points.append(p.new_position)
                velocity_points.append(p.new_velocity)
                acceleration_points.append(p.new_acceleration)
            way_index_list.append(points_index)
            points_index = self._accumulate_standing_time(0, points_index, time_points, position_points, velocity_points, acceleration_points)
        ##################################do speed ratio###############################################
        if speed_ratio >= 1:
            speed_ratio = 1
        elif speed_ratio >= 0.5:
            speed_ratio = 0.5
        elif speed_ratio >= 0.2:
            speed_ratio = 0.2
        else:
            speed_ratio = 0.1
        sample_interval = int(self._time_step / mini_ts * speed_ratio)
        new_way_index_list = np.round(np.array(way_index_list) / sample_interval).tolist()
        new_time_points = np.array(time_points[::sample_interval]) / speed_ratio
        new_position_points = position_points[::sample_interval]
        new_velocity_points = np.array(velocity_points[::sample_interval])* speed_ratio
        new_acceleration_points = np.array(acceleration_points[::sample_interval]) * speed_ratio * speed_ratio
        # assign last point to make end of trajectory precise
        points_index += 1
        new_time_points = np.append(new_time_points, new_time_points[-1] + self._time_step)
        new_position_points = np.append(new_position_points, [self._way_points[-1]], axis=0)
        new_velocity_points = np.append(new_velocity_points, [(new_position_points[-1] - new_position_points[-2]) / self._time_step], axis=0)
        new_acceleration_points = np.append(new_acceleration_points, [(new_velocity_points[-1] - new_velocity_points[-2]) / self._time_step], axis=0)
        new_way_index_list[-1] = len(new_acceleration_points) - 1
        # estimate jerk points
        new_jerk_points = np.gradient(new_acceleration_points, self._time_step, axis=0)
        csv_trajecoty = self._information2csv_trajectory(new_time_points, new_position_points, new_velocity_points, new_acceleration_points, new_jerk_points, 1)
        return csv_trajecoty, new_way_index_list
        