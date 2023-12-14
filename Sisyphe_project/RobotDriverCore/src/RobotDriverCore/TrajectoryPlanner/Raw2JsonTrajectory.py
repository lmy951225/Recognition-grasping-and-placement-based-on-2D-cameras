from typing import List, Union, Dict
from .TrajectoryAlgorithms.MetaTrajectoryTOPPRA import MetaTrajectoryTOPPRA
from ..RobotDriverCoreUtils.RobotConstraints import RobotConstraints

class Raw2JsonTrajectory():
    def __init__(self, traj_waypoints: List[List[float]], shot_flags: List[bool], offset_values: List[int], addr: List[List[int]]) -> None:
        self.__traj_waypoints = traj_waypoints
        self.__shot_flags = shot_flags
        self.__offset_values = offset_values
        self.__addr = addr
        self.__point_states = list()
        for addr_info, shot_flag in zip(self.__addr, self.__shot_flags):
            if shot_flag:
                self.__point_states.append("|".join([str(i) for i in addr_info]))
            else:
                self.__point_states.append(-1)
        self.__transition_waypoints: List[List[float]] = list()
        self.__transition_state: List[Union[str, int]] = list()

    def generate_transition_points(self, robot_constraints: RobotConstraints) -> int:
        meta_trajectory = MetaTrajectoryTOPPRA(way_points_or_toppra_instance=self.__traj_waypoints, trajectory_constraints=robot_constraints)
        # check traj way points first
        position_check = meta_trajectory.check_position_limit()
        if position_check != -1:
            return position_check
        # add transition if necessary
        transition_result = meta_trajectory.generate_transition_points(point_states=self.__point_states)
        if type(transition_result) == int:
            return transition_result
        #
        self.__transition_waypoints, self.__transition_state = transition_result
        return -1

    def generate_json_trajectory(self, trajectory_name: str, speed_coefficients: List[float] = None, standing_times: List[float] = None) -> Dict[str, Dict]:
        robot_config = {"use_do": True, "io_addr": [7, 8], "io_keep_cycles": 16, "acc_limit": 1, "jerk_limit": 1}
        speed_coefficients = speed_coefficients if speed_coefficients else [1] * len(self.__traj_waypoints)
        standing_times = standing_times if standing_times else [0] * len(self.__traj_waypoints)
        raw_trajectory_info = {
            "traj_waypoints": self.__traj_waypoints,
            "speed_coefficients": speed_coefficients,
            "standing_times": standing_times,
            "shot_flags": self.__shot_flags,
            "offset_values": self.__offset_values,
            "addr": self.__addr,
            "trajectory_transition_waypoints": self.__transition_waypoints,
            "trajectory_transition_states": self.__transition_state
        } 
        whole_trajectory = {
            "robot": robot_config,
            "flying_shots": {trajectory_name:raw_trajectory_info}
        }
        return whole_trajectory