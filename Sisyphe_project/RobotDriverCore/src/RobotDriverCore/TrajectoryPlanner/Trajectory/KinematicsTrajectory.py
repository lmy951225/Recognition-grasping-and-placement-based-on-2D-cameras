from typing import List
from .Trajectory import Trajectory

class KinematicsTrajectory(Trajectory):
    def __init__(self, trajectory_name: str, trajectory_path: str) -> None:
        super().__init__(trajectory_name, trajectory_path)
        self.__joint_vel_coef = self._trajectory_content["joint_vel_coef"]
        self.__joint_acc_coef = self._trajectory_content["joint_acc_coef"]
        self.__joint_jerk_coef = self._trajectory_content["joint_jerk_coef"]
        self.__traj_planning_time = self._trajectory_content["traj_planning_time"]
    
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
