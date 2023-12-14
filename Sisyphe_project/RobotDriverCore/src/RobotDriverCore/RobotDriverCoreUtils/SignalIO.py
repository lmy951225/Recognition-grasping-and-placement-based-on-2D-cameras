from aenum import Enum

class SignalIOType(Enum):
    SIGNAL_NO_IO=(0, 0)
    SIGNAL_DI = (1 , 1)
    SIGNAL_DO = (2 , 2)
    SIGNAL_UI = (3 ,20)
    SIGNAL_UO = (4 ,21)
    SIGNAL_RI = (5 , 8)
    SIGNAL_RO = (6 , 9)
    SIGNAL_GI = (7 ,-1)
    SIGNAL_GO = (8 ,-1)
    SIGNAL_SI = (-1,11)
    SIGNAL_SO = (-1,12)
    SIGNAL_WI = (-1,16)
    SIGNAL_WO = (-1,17)
    SIGNAL_WSI= (-1,26)
    SIGNAL_WSO= (-1,27)
    SIGNAL_F  = (-1,35)
    SIGNAL_M  = (-1,36)

    @property
    def Agilebot_code(self):
        return self.value[0]
    
    @property
    def Fanuc_code(self):
        return self.value[1]




