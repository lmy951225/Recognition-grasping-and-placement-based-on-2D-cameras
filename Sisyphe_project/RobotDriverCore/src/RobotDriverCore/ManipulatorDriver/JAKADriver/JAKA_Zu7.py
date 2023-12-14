import numpy as np

from .JAKADriver import JAKADriver
from RobotDriverCore.RobotDriverCoreUtils.ResponseStatus import ResponseStatus
from RobotDriverCore.RobotDriverCoreUtils.RobotConstraints import RobotConstraints
class JAKA_Zu7(JAKADriver):
    def __init__(self) -> None:
        super().__init__()

    def _generate_robot_constraints(self) -> RobotConstraints:
        """JAKA constraints is from JAKA website

        Returns:
            RobotConstraints: constraints of Agilebt P7A-900
        """
        robot_constraint = RobotConstraints() # unit is radian
        robot_constraint.time_step = 0.001
        # different limitS for                           J1     J2      J3      J4      J5      J6
        robot_constraint.max_position       =np.asarray([ 6.26, 4.6 , 2.85,    4.6,  6.26,  6.26])
        robot_constraint.min_position       =np.asarray([-6.26,-1.46,-2.85,  -1.46, -6.26, -6.26])
        robot_constraint.velocity_limit     =np.asarray([ 3.12, 3.12, 3.12,   3.12,  3.12,  3.12])
        robot_constraint.acceleration_limit =np.asarray([   10,   10,   10,     10,    10,    10])# guess parameters
        robot_constraint.jerk_limit         =np.asarray([   50,   50,   50,     50,    50,    50])# guess parameters
        return robot_constraint
        