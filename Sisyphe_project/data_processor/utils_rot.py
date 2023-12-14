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


def get_T_from_6dof(dof):
    trans=np.array(dof[:3])
    theta=np.array(dof[3:])
    R=eulerAngles2rotationMat(theta)
    T=np.concatenate([np.concatenate([R,trans[:,None]],axis=-1),\
                    np.array([[0,0,0,1]])],axis=0)
    return T

def get_6dof_from_T(T):

    R=T[:3,:3]
    trans=T[:3,3]
    theta=rotationMatrixToEulerAngles(R)
    out_6dof=list(trans)+list(theta)
    return np.array(out_6dof)

def rotation_matrix_to_quaternion(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def transform_matrix_to_pose(transform_matrix):

    translation = transform_matrix[:3, 3]
    rotation = transform_matrix[:3, :3]
    r_w, r_x, r_y, r_z = rotation_matrix_to_quaternion(rotation)
    pose = np.array([r_w, r_x, r_y, r_z, translation[0]/1000, translation[1]/1000, translation[2]/1000])

    return pose