from .AgilebotDriver import AgilebotDriver, AgilebotStatus2RobotDriverResponse
from Agilebot.IR.A.status_code import StatusCodeEnum
from RobotDriverCore.RobotDriverCoreUtils.ResponseStatus import ResponseStatus

class Agilebot_GBT_P7A_900(AgilebotDriver):
    def __init__(self) -> None:
        super().__init__()

    @AgilebotStatus2RobotDriverResponse
    def connect(self, controller_ip: str) -> ResponseStatus:
        connect_status = super().connect(controller_ip)
        if connect_status != StatusCodeEnum.OK:
            return connect_status
        self._flyshot.sync_fly_shot_config(controller_ip) # default is P7A-900, so failure doesn't matter
        return StatusCodeEnum.OK
