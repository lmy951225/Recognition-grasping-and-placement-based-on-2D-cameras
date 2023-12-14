import numpy as np
from RobotDriverCore.RobotDriverCoreUtils.RobotConstraints import RobotConstraints
from .AgilebotDriver import AgilebotDriver, AgilebotStatus2RobotDriverResponse
from Agilebot.IR.A.status_code import StatusCodeEnum
from RobotDriverCore.RobotDriverCoreUtils.ResponseStatus import ResponseStatus

class Agilebot_GBT_P20A_1800(AgilebotDriver):
    def __init__(self) -> None:
        super().__init__()

    @AgilebotStatus2RobotDriverResponse
    def connect(self, controller_ip: str) -> ResponseStatus:
        connect_status = super().connect(controller_ip)
        if connect_status != StatusCodeEnum.OK:
            return connect_status
        return self._flyshot.sync_fly_shot_config(controller_ip) # sync must successfully
    
    def _generate_robot_constraints(self) -> RobotConstraints:
        """Agilebot constraints is from Agilebot company 

        Returns:
            RobotConstraints: constraints of Agilebt GBT-P20A-1800
        """
        robot_constraint = RobotConstraints() # unit is radian
        robot_constraint.time_step = 0.001
        # different limitS for                           J1     J2      J3      J4      J5      J6
        robot_constraint.max_position       =np.asarray([ 3.19, 1.72, 2.77,   3.47,  2.37,  7.83])
        robot_constraint.min_position       =np.asarray([-3.19, -2.7,-1.43,  -3.47, -2.37, -7.83])
        robot_constraint.velocity_limit     =np.asarray([ 3.38, 2.89, 3.12,   6.97,  6.97, 10.45])
        robot_constraint.acceleration_limit =np.asarray([    5,    5,    8,     16,    18,    20])
        robot_constraint.jerk_limit         =np.asarray([    5,    5,    8,     16,    18,    20]) * 8
        return robot_constraint