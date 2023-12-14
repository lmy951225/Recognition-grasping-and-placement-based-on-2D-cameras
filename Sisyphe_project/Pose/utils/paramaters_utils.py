'''
Author      :XiaoZhiheng
Time        :2023/05/18 17:05:28
'''
# MachineTranslationType = "Fenxianhe"
# MachineTranslationType = "Geciban"
MachineTranslationType = "Mensuo"

RRTInterNum:int = 1500
RRTGoalSampleRate:int = 0.2
is_informed:bool = True

RRTLMS:float = 0.04  
#RRT linear move step, unit of radian, 0.01 radian approximately equal to 0.573 degree

CollisionCheckLMS:float = 0.04   
# CollisionCheck linear move step, unit of radian, 0.01 radian approximately equal to 
# 0.573 degree, if in foundation translation, the unit is meters.

ProcessesNum:int = 6

IKSolverType:str = "AN"  # LM or "AN"  ,求逆解的方式， LM表示迭代法， AN为解析法

SomWeight: float = 0.6  # Som 里xyz 和rxryrz的加权系数， 0.6表示两者系数为1：0.6



