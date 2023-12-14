from typing import List, Union
import numpy as np
import numpy.typing as npt


class RobotConstraints():
    """constraints of robot, including max/min joint positon, velocity limit, acceleration limit, jerk limit and time step
    if jerk limit is not considered, then set jerk limit infinit large would make it work
    """
    def __init__(self, dof: int = 6) -> None:
        self.__dof = dof
        self.__max_position = np.zeros(dof)
        self.__min_position = np.zeros(dof)
        self.__velocity_limit = np.zeros(dof)
        self.__acceleration_limit = np.zeros(dof)
        self.__jerk_limit = np.zeros(dof)
        self.__time_step = 0

    @property
    def degree_of_freedom(self) -> int:
        return self.__dof

    @property
    def time_step(self) -> float:
        return self.__time_step
    
    @time_step.setter
    def time_step(self, value: float):
        self.__time_step = value

    @property
    def max_position(self) -> npt.NDArray[np.float_]:
        return self.__max_position

    @max_position.setter
    def max_position(self, max_positions: npt.NDArray[np.float_]) -> None:
        self.__max_position = max_positions

    @property
    def min_position(self) -> npt.NDArray[np.float_]:
        return self.__min_position

    @min_position.setter
    def min_position(self, min_positions: npt.NDArray[np.float_]) -> None:
        self.__min_position = min_positions

    @property
    def velocity_limit(self) -> npt.NDArray[np.float_]:
        return self.__velocity_limit
    
    @velocity_limit.setter
    def velocity_limit(self, velocity_limits: npt.NDArray[np.float_]) -> None:
        self.__velocity_limit = velocity_limits

    @property
    def acceleration_limit(self) -> npt.NDArray[np.float_]:
        return self.__acceleration_limit
    
    @acceleration_limit.setter
    def acceleration_limit(self, acceleration_limits: npt.NDArray[np.float_]) -> None:
        self.__acceleration_limit = acceleration_limits

    @property
    def jerk_limit(self) -> npt.NDArray[np.float_]:
        return self.__jerk_limit
    
    @jerk_limit.setter
    def jerk_limit(self, jerk_limits: npt.NDArray[np.float_]) -> None:
        self.__jerk_limit = jerk_limits