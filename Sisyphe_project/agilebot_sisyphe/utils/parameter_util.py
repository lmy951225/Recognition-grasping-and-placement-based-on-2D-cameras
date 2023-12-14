import numpy as np
 # standard constraints from Manipulator Producer
'''
POSITION_CONSTRAINT = np.array([[-2.96, -2.35, -1.22, -3.31, -2.0, -6.28], 
                                [ 2.96,  1.74,  3.49,  3.31,  2.0,  6.28]])
VELOCITY_CONSTRAINT = np.array([5.81, 4.66, 5.81, 7.85, 7.07, 10.56])
ACCELERATION_CONSTRAINTS = np.array([24, 20, 24, 32, 28, 40])
JERK_CONSTRAINTS = np.array([240, 200, 240, 320, 280, 400])
'''
# calculate constraints for safety
POSITION_CONSTRAINT = np.array([[-2.94, -2.33, -1.2, -3.29, -1.98, -6.26], 
                                [ 2.94,  1.72,  3.47,  3.29,  1.98,  6.26]])
VELOCITY_CONSTRAINT = np.array([5.71, 4.56, 5.71, 7.75, 6.97, 10.46])
ACCELERATION_CONSTRAINTS = np.array([23, 19, 23, 31, 27, 39])
JERK_CONSTRAINTS = np.array([230, 190, 230, 310, 270, 390])
TIME_STEP = 0.001