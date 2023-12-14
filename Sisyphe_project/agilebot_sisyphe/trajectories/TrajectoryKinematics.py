from typing import List
import json
from trajectories.Trajectory import Trajectory

class TrajectoryKinematics(Trajectory):
    def __init__(self, trajectory_name: str, trajectory_path: str) -> None:
        super().__init__(trajectory_name, trajectory_path)
        with open(trajectory_path, "r") as f:
            trajectory_content = json.load(f)
        self.__joint_vel_coef = trajectory_content["joint_vel_coef"]
        self.__joint_acc_coef = trajectory_content["joint_acc_coef"]
        self.__joint_jerk_coef = trajectory_content["joint_jerk_coef"]
        self.__traj_planning_time = trajectory_content["traj_planning_time"]
    
    @property
    def joint_vel_coef(self) -> List[float]:
        return self.__joint_vel_coef
    
    @property
    def joint_acc_coef(self) -> List[float]:
        return self.__joint_acc_coef
    
    @property
    def joint_jerk_coef(self) -> List[float]:
        return self.__joint_jerk_coef
    
    @property
    def traj_planning_time(self) -> float:
        return self.__traj_planning_time
