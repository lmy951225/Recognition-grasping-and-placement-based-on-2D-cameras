import numpy as np
import  math
def rotationMatrixToEulerAngles(R):

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    # singular = sy < 1e-6
    # if  not singular:
    x = math.atan2(R[2,1] , R[2,2])
    y = math.atan2(-R[2,0], sy)
    z = math.atan2(R[1,0], R[0,0])
    # else :
    #     x = math.atan2(-R[1,2], R[1,1])
    #     y = math.atan2(-R[2,0], sy)
    #     z = 0
    #print('dst:', R)
    x = x*180.0/np.pi
    y = y*180.0/np.pi
    z = z*180.0/np.pi
    return x,y,z
# print(rot2eul(R))

# print(rotationMatrixToEulerAngles(R))

############



def eulerAngles2rotationMat(theta, format='degree'):
    """
    Calculates Rotation Matrix given euler angles.
    :param theta: 1-by-3 list [rx, ry, rz] angle in degree
    :return:
    RPY角，是ZYX欧拉角，依次 绕定轴XYZ转动[rx, ry, rz]
    """
    if format == 'degree':
        theta = [i * math.pi / 180.0 for i in theta]

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R
