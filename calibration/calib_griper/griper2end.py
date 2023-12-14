import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

def loadtxtmethod(filename):
    data = np.loadtxt(filename,dtype=np.float32,delimiter=',')
    return data

def xyz_rxryrz2transformation(xyz_rxryrz: np.ndarray): 
        transformation = np.identity(4)
        transformation[:3, :3] = R.from_euler(seq="xyz", angles=xyz_rxryrz[3:], degrees=True).as_matrix()
        transformation[0, 3] = xyz_rxryrz[0]
        transformation[1, 3] = xyz_rxryrz[1]
        transformation[2, 3] = xyz_rxryrz[2]
        return transformation

def transformation2xyz_rxryrz(transformation: np.ndarray):  
    rxryrz = R.from_matrix(transformation[:3, :3]).as_euler(seq="xyz", degrees=True)
    return np.concatenate([transformation[:3,3],rxryrz])

def Tcompute(RT1,RT2,RT3,RT4):
    R1 = RT1[:3,:3]
    R2 = RT2[:3,:3]
    R3 = RT3[:3,:3]
    R4 = RT4[:3,:3]
    T1 = RT1[:3,3].reshape(-1,1)
    T2 = RT2[:3,3].reshape(-1,1)
    T3 = RT3[:3,3].reshape(-1,1)
    T4 = RT4[:3,3].reshape(-1,1)
    R = np.vstack([R1-R2,R2-R3,R3-R4])
    T = np.vstack([T2-T1,T3-T2,T4-T3])
    RR = R.T@R
    P = np.linalg.inv(RR)@R.T@T
    return P

def Rcompute(RT1,RT2,RT3):
    X = RT2[:3,3] - RT1[:3,3]
    Z = RT3[:3,3] - RT2[:3,3]
    Y = np.cross(Z,X)
    Z = np.cross(X,Y)
    X = X / np.linalg.norm(X)
    Y = Y / np.linalg.norm(Y)
    Z = Z / np.linalg.norm(Z)
    R1 = np.column_stack((X,Y,Z))
    R2 = RT1[:3,:3]
    R = np.linalg.inv(R2)@R1
    return R

def main(filename,d):
    '''
    param:filename,存放六个点位的文件,txt
    param:d:夹爪夹指末端与中间的距离
    '''
    data = loadtxtmethod(filename)
    RT = []
    for idx in range(len(data)):
        RT.append(xyz_rxryrz2transformation(data[idx]))
    P = Tcompute(RT[0],RT[1],RT[2],RT[3])
    P[2] = P[2] - d #上调至夹爪中心
    R1 = Rcompute(RT[3],RT[4],RT[5])
    RT = np.column_stack([R1, P])  # 列合并
    RT = np.row_stack((RT, np.array([0,0,0,1])))
    return RT
      

if __name__== '__main__':
    #二指夹爪的标定
    filename = './griper2.txt' #存放6个点的位姿
    d = 5 #夹爪闭合后末端与夹指中心的距离，根据夹爪型号设定
    RT2 = main(filename,d)
    print(f'二指夹爪到tcp转换矩阵为:\n{RT2}')

    #三指夹爪的标定
    filename = './griper3.txt'
    d = 7
    RT3 = main(filename,d)
    print(f'三指夹爪到tcp转换矩阵为:\n{RT3}')

