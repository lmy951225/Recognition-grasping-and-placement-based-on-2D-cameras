from .FanucDriver import FanucDriver
from RobotDriverCore.RobotDriverCoreUtils.ResponseStatus import ResponseStatus

class Fanuc_LR_Mate_200iD_7L(FanucDriver):
    def __init__(self, is_ROBOTGUIDE: bool = True) -> None:
        super().__init__(is_ROBOTGUIDE)
