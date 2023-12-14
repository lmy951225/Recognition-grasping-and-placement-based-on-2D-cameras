from typing import List, cast
from Agilebot.IR.A.arm import Arm
from Agilebot.IR.A.sdk_classes import DHparam
from Agilebot.IR.A.status_code import StatusCodeEnum

class ArmSingleton(Arm):
    def __init__(self):
        super().__init__()
        self.__speed: float = 1 # default speed is 1
    
    def __new__(cls): # for singleton 
        if not hasattr(cls, 'instance'):
            cls.instance = super(ArmSingleton, cls).__new__(cls)
        return cls.instance
    
    @property
    def is_connected(self) -> bool:
        return self.is_connect()
    
    @property
    def speed(self) -> float:
        return self.__speed
    
    @speed.setter
    def speed(self, value) -> float:
        if value > 0 and value <= 1:
            self.__speed = value
        else:
            raise ValueError("not support value, 0 < speed <= 1")
    
    def connect(self, arm_controller_ip: str) -> StatusCodeEnum:
        connect_results = super().connect(arm_controller_ip)
        return connect_results
    
    def disconnect(self):
        self.__speed = 1
        return super().disconnect()
    
    def get_DH_parameters(self) -> List[List[float]]:
        results = list()
        if self.is_connected:
            DH_parameters, getDH_status = self.motion.get_DH_param()
            if getDH_status == StatusCodeEnum.OK:
                DH_parameters = cast(List[DHparam], DH_parameters)
                for dh in DH_parameters:
                    results.append([dh.id, dh.d, dh.alpha, dh.offset])
        return results
            
    
# only import this instance
ArmInstance = ArmSingleton()

if __name__ == "__main__":
    print(ArmInstance.connect("172.16.1.202"))
    print(ArmInstance.disconnect())
    ArmInstance.speed = 0.5
