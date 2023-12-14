"""
@Author      :   XiaoZhiheng
@Time        :   2022/12/20 18:13:21
"""
import numpy as np
from math import pi

RobotParameters = {
    "nachi":{
        "DHP":np.array([
        [0.   ,     0, 0.345,     0],
        [0.050, -pi/2, 0.0,     0], 
        [0.330,     0, 0.   , -pi/2],
        [0.045, -pi/2, 0.   ,     0],
        [0.   ,  pi/2, 0.340,     0],
        [0.   , -pi/2, 0.000,     0], 
        [0.   ,     0, 0.063,     0]]),
        "LowerBoundary":np.asarray([0,0,0,0,0,0]),
        "UpperBoundary":np.asarray([0,0,0,0,0,0])
    },
    "LR_Mate_200iD_7L":{
        "DHP":np.array([
        [0.   ,     0, 0.33,     0],
        [0.050, -pi/2, 0.0,     0], 
        [0.440,     0, 0.   , -pi/2],
        [0.035, -pi/2, 0.   ,     0],
        [0.   ,  pi/2, 0.420,     0],
        [0.   , -pi/2, 0.000,     0], 
        [0.   ,     0, 0.080,     0]]),
        "LowerBoundary":np.array([-90, -97, -70, -150, -97, -360]),
        "UpperBoundary":np.array([ 90, 90, 65,  150,  97,  0 ])
    },
    "LR_Mate_200iD":{
        "DHP":np.array([
        [0.   ,     0, 0.33,     0],
        [0.050, -pi/2, 0,     0], 
        [0.330,     0, 0.   , -pi/2],
        [0.035, -pi/2, 0.   ,     0],
        [0.   ,  pi/2, 0.335,     0],
        [0.   , -pi/2, 0.000,     0], 
        [0.   ,     0, 0.080,     0]]),
        "LowerBoundary":np.array([-90, -97, -70, -150, -97, -360]),
        "UpperBoundary":np.array([ 90, 90, 65,  150,  97,  0  ])
    },
    
    "P7A_900":{
        "DHP":np.array([
        [0.   ,     0, 0.1785,     0],
        [0.030, -pi/2, 0.1865,     0], 
        [0.45,     0, 0.   , -pi/2],
        [0.035, -pi/2, 0.0 ,     0],
        # [0.   ,  pi/2, 0.2285+0.1915,     0],
        [0.   ,  pi/2, 0.420,     0],
        [0.   , -pi/2, 0.000,     0], 
        [0.   ,     0, 0.102,     0]]),
        "LowerBoundary":np.array([-90, -97, -65, -150, -97, -180]),
        "UpperBoundary":np.array([ 90, 90, 65,  150,  97,  180  ])
    },

}


