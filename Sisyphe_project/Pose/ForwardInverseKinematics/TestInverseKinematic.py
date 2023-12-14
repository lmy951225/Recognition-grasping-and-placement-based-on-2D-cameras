import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from utils.tools import real2virtual, homogeneous_matrix_to_euler
from ForwardInverseKinematics.RobotForwardKinematics import RobotForwardKinematics
from ForwardInverseKinematics.RobotAnalyticalInverseKinematics import RobotAnalyticalInverseKinematics
from ForwardInverseKinematics.RobotLevenbergMarquardtInverseKinematics import RobotLevenbergMarquardtInverseKinematics
from ForwardInverseKinematics.robotParameters import RobotParameters
import numpy as np 
from time import time 
import pandas as pd 
from utils.tools import virtual2real, real2virtual
from utils.home_joints import HomeJoints

def random_generate_thetas(lower, upper):
    
    rand = np.random.random(6)
    return rand*(upper - lower) + lower
    
manipulator_type = "LR_Mate_200iD_7L"
FK = RobotForwardKinematics(manipulator_type)

LMIK = RobotLevenbergMarquardtInverseKinematics(manipulator_type)
ANIK = RobotAnalyticalInverseKinematics(manipulator_type)

LowerBoundary = RobotParameters[manipulator_type]["LowerBoundary"]
UppererBoundary = RobotParameters[manipulator_type]["UpperBoundary"]

random_thetas_list = []
LM_ins_thetas_list = []
AN_ins_thetas_list = []

LM_l2_error_list=[]
AN_l2_error_list=[]
LM_time_list = []
AN_time_list = []
LM_is_q_list = []
AN_is_q_list = []
for i in range(100):
    random_thetas = real2virtual(
        random_generate_thetas(LowerBoundary, UppererBoundary),
        manipulator_type, True
    )
    target_matrix = FK.forward_kinematics(random_thetas)
    
    target_matrix = np.array([[-0.406577, -0.499998,  0.764655,  0.627455],
 [-0.234736,  0.866027,  0.441471,  0.016774],
 [-0.882946, -0. ,      -0.469474,  0.624264],
 [ 0.    ,    0.     ,   0.    ,    1.      ]])
    # print(random_thetas)
    # xyzrxryrz = homogeneous_matrix_to_euler(target_matrix)
    # xyzrxryrz[0:3] *= 1000
    
    start_thetas = HomeJoints
    start_thetas = real2virtual(start_thetas, "LR_Mate_200iD_7L", isToRadians=True)
    
    
    t1 = time()
    LM_ins_thetas, LM_q = LMIK.inverse_kinematics(target_matrix, start_thetas)
    t2 = time()
    # print(LM_ins_thetas)

    AN_ins_thetas, AN_q = ANIK.inverse_kinematics(target_matrix,start_thetas)
    t3 = time()
    print(random_thetas)
    print(LM_ins_thetas)
    print(AN_ins_thetas)
    

    # for thetas in AN_ins_thetas:
    if AN_ins_thetas is not None:
        AN_m = FK.forward_kinematics(AN_ins_thetas)
        AN_l2_error = np.sum((AN_m-target_matrix) ** 2) ** 0.5
    else:
        AN_l2_error = "inf"
    if LM_ins_thetas is not None:
        LM_m = FK.forward_kinematics(LM_ins_thetas)
        LM_l2_error = np.sum((LM_m-target_matrix) ** 2) ** 0.5
    else:
        LM_l2_error = "inf"
   
    random_thetas_list.append(str(random_thetas))
    LM_ins_thetas_list.append(str(LM_ins_thetas))
    AN_ins_thetas_list.append(str(AN_ins_thetas))
    LM_l2_error_list.append(LM_l2_error)
    AN_l2_error_list.append(AN_l2_error)
    
    LM_time_list.append(t2-t1)
    AN_time_list.append(t3-t2)
    
    LM_is_q_list.append(LM_q) 
    AN_is_q_list.append(AN_q)
    
cols = ["random_theta", "LM_ins_thetas", "AN_ins_thetas", "LM_l2_error", "AN_l2_error", "LM_time(s)", "AN_time(s)", "LM_is_q", "AN_is_q"]    
data = pd.concat([
    pd.DataFrame(random_thetas_list, columns=[cols[0]]),
    pd.DataFrame(LM_ins_thetas_list, columns=[cols[1]]),
    pd.DataFrame(AN_ins_thetas_list, columns=[cols[2]]),
        pd.DataFrame(LM_l2_error_list, columns=[cols[3]]),
        pd.DataFrame(AN_l2_error_list, columns=[cols[4]]),
        pd.DataFrame(LM_time_list, columns=[cols[5]]),
        pd.DataFrame(AN_time_list, columns=[cols[6]]),
        pd.DataFrame(LM_is_q_list, columns=[cols[7]]),
        pd.DataFrame(AN_is_q_list, columns=[cols[8]])
        ], axis=1)
data.to_csv("data.csv")
print(data)
        
        



