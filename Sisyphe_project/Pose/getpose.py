import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import pandas as pd
import matplotlib.pyplot as plt
import sys
from IK import getJoint
import random
from mesh_subdivision_filter import RT_tcp_sort
# import pybullet as p 
np.set_printoptions(suppress=True)
# RT_tcp = np.load('/home/adt/lingji_lite_scripts/RT_tcp.npy')
def transformation2xyz_rxryrz(transformation: np.ndarray):  # only work for fannuc
        rxryrz = R.from_matrix(transformation[:3, :3]).as_euler(seq="xyz", degrees=True)
        # for i in range(0,3): #确保角度不超过180
        #     if abs(180 - abs(rxryrz[i])) < 0.1:
        #         rxryrz[i] = 179.9
        return np.concatenate([transformation[:3,3],rxryrz])

def getTcppoint(rt_tcp):
    rt_tcp_euler = []
    for idx in range(len(rt_tcp)):
        rt_tcp_euler.append(transformation2xyz_rxryrz(rt_tcp[idx]))
    c = getJoint()
    p_xyzrxryrz = list()
    p_joint = list()
    for k in range(len(rt_tcp_euler)):
        if np.any(c.transferCartesian2Joints(rt_tcp_euler[k]) !=  -99999 ):
            p_joint.append(c.transferCartesian2Joints(rt_tcp_euler[k]))
            p_xyzrxryrz.append(rt_tcp_euler[k])
        # else:
        #     print(f'第{k}个点无解')
    print(f'理论插补点个数:{len(rt_tcp)}----实际插补点个数:{len(p_xyzrxryrz)}----实际逆解个数:{len(p_joint)}')
    # p_all = p_xyzrxryrz + p_joint
    # df = pd.DataFrame(p_all)
    # output = df.to_string(index=False, header=False)
    # print(output)
    return p_xyzrxryrz,p_joint

def getCircleTraj(P,n,z,x):
    """
    param:P:圆心+初始点姿态角(x,y,z,rx,ry,rz)
    param:z:物体中心高度
    param:x:相机与tcp的x向距离
    param:n:插值点数
    """
    point_xyz = []
    m = (2*np.pi)/n
    r = (P[2]-z) / np.tan((P[4]+90) * np.pi / 180) + x/np.sin((P[4]+90) * np.pi / 180)
    # r_cam = P[2] / np.sin((P[4]+90) * np.pi / 180) - 190.8758467 + x/np.tan((P[4]+90) * np.pi / 180)
    # print(r_cam)
    P0 = [0,0,0,P[3],P[4],P[5]]
    P1 = [0,0,0,P[3],P[4],P[5]+180]
    P2 = [0,0,0,P[3],P[4],P[5]-180]
    j = int(n/2 + 1)
    rxyz_0_180 = Pose_slerp(P0,P1,j)
    rxyz_180_0 = Pose_slerp(P2,P0,j)
    rxyz = rxyz_0_180.tolist() + rxyz_180_0[1:-1].tolist()
    for i in range(n):
        x = P[0] - r*np.cos(m*i)
        y = P[1] - r*np.sin(m*i)
        point_xyz.append(np.array([x,y,P[2],rxyz[i][0],rxyz[i][1],rxyz[i][2]]))
    c = getJoint()
    p_xyzrxryrz = list()
    p_joint = list()
    # print(f'len{len(p_xyzrxryrz)}')
    for k in range(len(point_xyz)):
        if np.any(c.transferCartesian2Joints(point_xyz[k]) !=  -99999 ):
            p_joint.append(c.transferCartesian2Joints(point_xyz[k]))
            p_xyzrxryrz.append(point_xyz[k])
        else:
            print(f'第{k}个点无解')
    print(f'理论插补点个数:{len(point_xyz)}----实际插补点个数:{len(p_xyzrxryrz)}----实际逆解个数:{len(p_joint)}')
    # p_all = p_xyzrxryrz + p_joint
    # df = pd.DataFrame(p_all)
    # output = df.to_string(index=False, header=False)
    # print(output)
    return p_xyzrxryrz,p_joint,P

#求向量旋转过程的旋转矩阵3*3
def rotation_matrix(axis,theta):
    axis = axis / np.linalg.norm(axis)
    # print(axis)
    a = np.cos(theta / 2)
    bcd = -axis * np.sin(theta / 2)
    b = bcd[0]
    c = bcd[1]
    d = bcd[2]
    # print(a,b,c,d)
    x0=(a*a+b*b-c*c-d*d);x1=2*(b*c+a*d);x2=2*(b*d-a*c)
    y0=2*(b*c-a*d);y1=a*a+c*c-b*b-d*d;y2=2*(c*d+a*b)
    z0=2*(b*d+a*c);z1=2*(c*d-a*b);z2=a*a+d*d-b*b-c*c
    RT_1 = np.array([[x0,x1,x2],[y0,y1,y2],[z0,z1,z2]])
    # print(RT_1)
    return RT_1

def get_circle(P1,P2,P3):
    '''
    param P1 P2 P3:空间不共线的三个点的坐标
    return Oc:拟合圆的坐标
    '''
    a1:float = P1[1]*P2[2] - P2[1]*P1[2] - P1[1]*P3[2] + P3[1]*P1[2] + P2[1]*P3[2] - P3[1]*P2[2]
    b1:float = -(P1[0]*P2[2] - P2[0]*P1[2] - P1[0]*P3[2] + P3[0]*P1[2] + P2[0]*P3[2] -P3[0]*P2[2])
    c1:float = P1[0]*P2[1] - P2[0]*P1[1] - P1[0]*P3[1] + P3[0]*P1[1] + P2[0]*P3[1] - P3[0]*P2[1]
    d1:float = -(P1[0]*P2[1]*P3[2] - P1[0]*P3[1]*P2[2] - P2[0]*P1[1]*P3[2] + P2[0]*P3[1]*P1[2] + P3[0]*P1[1]*P2[2] - P3[0]*P2[1]*P1[2])

    a2:float = 2*(P2[0] - P1[0])
    b2:float = 2*(P2[1] - P1[1])
    c2:float = 2*(P2[2] - P1[2])
    d2:float = P1[0]*P1[0] + P1[1]*P1[1] + P1[2]*P1[2] - P2[0]*P2[0] -P2[1]*P2[1] - P2[2]*P2[2]

    a3:float = 2*(P3[0] - P1[0])
    b3:float = 2*(P3[1] - P1[1])
    c3:float = 2*(P3[2] - P1[2])
    d3:float = P1[0]*P1[0] + P1[1]*P1[1] + P1[2]*P1[2] - P3[0]*P3[0] -P3[1]*P3[1] - P3[2]*P3[2]

    a:float = -(b1*c2*d3 - b1*c3*d2 - b2*c1*d3 + b2*c3*d1 + b3*c1*d2 - b3*c2*d1) / (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)
    b:float = (a1*c2*d3 - a1*c3*d2 - a2*c1*d3 + a2*c3*d1 + a3*c1*d2 - a3*c2*d1) / (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)
    c:float = -(a1*b2*d3 - a1*b3*d2 - a2*b1*d3 + a2*b3*d1 + a3*b1*d2 - a3*b2*d1) / (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)

    Oc = np.array([a,b,c])
    # print("拟合圆心坐标Oc:\n",Oc)
    return Oc

#利用Slerp插值法进行姿态的平滑插值
def Pose_slerp(P1,P2,n):
    key_rots = np.array([[P1[3],P1[4],P1[5]],[P2[3],P2[4],P2[5]]])
    key_rots = R.from_euler('xyz',key_rots,degrees=True)
    key_times = [0,1]
    slerp = Slerp(key_times,key_rots)
    times = np.linspace(0,1,n)
    # print(f'times{times}')
    interp_rots = slerp(times)
    rxryrz = interp_rots.as_euler('xyz',degrees=True)
    # print(rxryrz[0])
    return rxryrz

#求圆弧轨迹插补点位姿
def SpaceCirclefor4(P1,P2,P3,P4,n):
    #求姿态
    Rots = Pose_slerp(P1,P2,n)
    Rots2 = Pose_slerp(P2,P3,n)
    Rots3 = Pose_slerp(P3,P4,n)
    Rots4 = Pose_slerp(P4,P1,n)
    #求圆心
    p1 = np.array([P1[0],P1[1],P1[2]])
    p2 = np.array([P2[0],P2[1],P2[2]])
    p3 = np.array([P3[0],P3[1],P3[2]])
    p4 = np.array([P4[0],P4[1],P4[2]])
    P = get_circle(p1,p2,p3)
    P2 = get_circle(p3,p4,p1)
    #求末端执行器在圆弧上的运动轴,过圆心且垂直于圆平面
    vector_start_big = p1 - P
    vector_middle_big = p2 - P

    vector_start_big2 = p3 - P2
    vector_middle_big2 = p4 - P2

    vector_start = (p1 - P) / np.linalg.norm(p1 - P)
    # print(vector_start_big,vector_start)
    vector_middle = (p2 - P) / np.linalg.norm(p2 - P)
    vector_final = (p3 - P) / np.linalg.norm(p3 - P)


    vector_start2 = (p3 - P2) / np.linalg.norm(p3 - P2)
    # print(vector_start_big,vector_start)
    vector_middle2 = (p4 - P2) / np.linalg.norm(p4 - P2)
    vector_final2 = (p1 - P2) / np.linalg.norm(p1 - P2)

    rotation_axis = np.cross(vector_start,vector_middle)
    rotation_axis2 = np.cross(vector_middle,vector_final)
    rotation_axis3 = np.cross(vector_start2,vector_middle2)
    rotation_axis4 = np.cross(vector_middle2,vector_final2)
    # print(rotation_axis)
    #生成轨迹
    theta = np.arccos(np.dot(vector_start,vector_middle))
    theta2 = np.arccos(np.dot(vector_middle,vector_final))
    theta3 = np.arccos(np.dot(vector_start2,vector_middle2))
    theta4 = np.arccos(np.dot(vector_middle2,vector_final2))

    print(f'弧线1旋转角度为{theta*180/np.pi}')
    print(f'弧线2旋转角度为{theta2*180/np.pi}')
    print(f'弧线3旋转角度为{theta3*180/np.pi}')
    print(f'弧线4旋转角度为{theta4*180/np.pi}')

    theta_per = theta / (n-1)
    theta_per2 = theta2 / (n-1)
    theta_per3 = theta3 / (n-1)
    theta_per4 = theta4 / (n-1)

    theta_current = 0
    theta_current2 = 0
    theta_current3 = 0
    theta_current4 = 0

    p_xyz = list()
    p_xyz2 = list()
    p_xyz3 = list()
    p_xyz4 = list()
    p_joint = list()
    for i in range(1,n+1):
        matrix_current = rotation_matrix(rotation_axis,theta_current)
        matrix_current2 = rotation_matrix(rotation_axis2,theta_current2)
        matrix_current3 = rotation_matrix(rotation_axis3,theta_current3)
        matrix_current4 = rotation_matrix(rotation_axis4,theta_current4)

        vector_current = np.dot(matrix_current,vector_start_big.T)
        vector_current2 = np.dot(matrix_current2,vector_middle_big.T)
        vector_current3 = np.dot(matrix_current3,vector_start_big2.T)
        vector_current4 = np.dot(matrix_current4,vector_middle_big2.T)
        
        p_current = P + vector_current.T
        p_current2 = P + vector_current2.T
        p_current3 = P2 + vector_current3.T
        p_current4 = P2 + vector_current4.T

        p_xyz.append(np.hstack([p_current,Rots[i-1]]))
        p_xyz2.append(np.hstack([p_current2,Rots2[i-1]]))
        p_xyz3.append(np.hstack([p_current3,Rots3[i-1]]))
        p_xyz4.append(np.hstack([p_current4,Rots4[i-1]]))

        theta_current = i * theta_per
        theta_current2 = i * theta_per2
        theta_current3 = i * theta_per3
        theta_current4 = i * theta_per4
    c = getJoint()
    p_final = p_xyz + p_xyz2[1:] + p_xyz3[1:] + p_xyz4[1:-1]
    p_xyzrxryrz = list()
    # print(f'len{len(p_xyzrxryrz)}')
    for j in range(len(p_final)):
        if np.any(c.transferCartesian2Joints(p_final[j]) !=  -99999 ):
            p_joint.append(c.transferCartesian2Joints(p_final[j]))
            p_xyzrxryrz.append(p_final[j])
        else:
            print(f'第{j}个点无解')
    print(f'理论插补点个数:{len(p_final)}----实际插补点个数:{len(p_xyzrxryrz)}----实际逆解个数:{len(p_joint)}')
    # p_all = p_xyzrxryrz + p_joint
    # df = pd.DataFrame(p_all)
    # output = df.to_string(index=False, header=False)
    # print(output)
    return p_xyzrxryrz,P,P2

def SpaceCirclefor3(P1,P2,P3,n):
    #求插补姿态
    Rots = Pose_slerp(P1,P3,n)
    #求圆心
    p1 = np.array([P1[0],P1[1],P1[2]])
    p2 = np.array([P2[0],P2[1],P2[2]])
    p3 = np.array([P3[0],P3[1],P3[2]])
    P = get_circle(p1,p2,p3)
    #求末端执行器在圆弧上的运动轴,过圆心且垂直于圆平面
    vector_start_big = p1 - P
    vector_start = (p1 - P) / np.linalg.norm(p1 - P)
    vector_final = (p3 - P) / np.linalg.norm(p3 - P)
    rotation_axis = np.cross(vector_start,vector_final)
    #生成轨迹
    theta = np.arccos(np.dot(vector_start,vector_final))
    # print(f'弧线旋转角度为{theta*180/np.pi}')
    
    theta_per = theta / (n-1)
    theta_current = 0
   

    p_xyz = list()
    p_joint = list()

    for i in range(1,n+1):
        matrix_current = rotation_matrix(rotation_axis,theta_current)
        vector_current = np.dot(matrix_current,vector_start_big.T)
        p_current = P + vector_current.T
        p_xyz.append(np.hstack([p_current,Rots[i-1]]))
        theta_current = i * theta_per

    p_xyzrxryrz = list()
    # print(f'len{len(p_xyzrxryrz)}')
    c = getJoint()
    for j in range(len(p_xyz)):
        if np.any(c.transferCartesian2Joints(p_xyz[j]) !=  -99999 ):
            p_joint.append(c.transferCartesian2Joints(p_xyz[j]))
            p_xyzrxryrz.append(p_xyz[j])
        else:
            print(f"第{j}个点无解")
    print(f'理论插补点个数:{len(p_xyz)}----实际插补点个数:{len(p_xyzrxryrz)}----实际逆解个数:{len(p_joint)}')
    # p_all = p_xyzrxryrz + p_joint
    # p_circle = p_xyz + p_xyz2[1:] + p_xyz3[1:-1]
    # df = pd.DataFrame(p_all)
    # output = df.to_string(header=None,index=None)
    # print(output)
    return p_xyzrxryrz,P

def RandData(P,n,m1,m2):
    '''
    param:P:指定tcp位姿 1x6
    param:n:随机个数
    param:m1:位置插补范围[-m1,m1]mm
    param:m2:姿态插补范围[-m2,m2]deg
    '''
    # np.random.seed(123)
    a = random.Random()
    a.seed(3)
    A = np.array([a.uniform(-m1,m1) for i in range(n)])
    B = np.array([a.uniform(-m1,m1) for i in range(n)])
    C = np.array([a.uniform(-m1,m1) for i in range(n)])
    D = np.array([a.uniform(-m2,m2) for i in range(n)])
    E = np.array([a.uniform(-m2,m2) for i in range(n)])
    F = np.array([a.uniform(-m2*10,m2*10) for i in range(n)])
    Data = []
    for j in range(n):
        Data.append(np.array([P[0]+A[j],P[1]+B[j],P[2]+C[j],P[3]+D[j],P[4]+E[j],P[5]+F[j]]))
    c = getJoint()
    p_xyzrxryrz = list()
    p_joint = list()
    # print(f'len{len(p_xyzrxryrz)}')
    for k in range(len(Data)):
        if np.any(c.transferCartesian2Joints(Data[k]) !=  -99999 ):
            p_joint.append(c.transferCartesian2Joints(Data[k]))
            p_xyzrxryrz.append(Data[k])
    print(f'理论插补点个数:{len(Data)}----实际插补点个数:{len(p_xyzrxryrz)}----实际逆解个数:{len(p_joint)}')
    # p_all = p_xyzrxryrz + p_joint
    # df = pd.DataFrame(p_all)
    # output = df.to_string(index=False, header=False)
    # print(output)
    return p_xyzrxryrz

if __name__ == '__main__':

    # arguments = sys.argv[1:]

    # if len(arguments) != 4:
    #     print("Error: Please provide 4 lists of 6 doubles each.", file=sys.stderr)
    #     sys.exit(1)

    # try:

    #     P1 = [float(x) for x in arguments[0].split(',')]
    #     P2 = [float(x) for x in arguments[1].split(',')]
    #     P3 = [float(x) for x in arguments[2].split(',')]
    #     P4 = [float(x) for x in arguments[3].split(',')]

    #     if len(P1) != 6 or len(P2) != 6 or len(P3) != 6 or len(P4) != 6:
    #         print("Error: Each list should contain 6 doubles.", file=sys.stderr)
    #         sys.exit(1)

    # except ValueError:
    #     print("Error: Invalid input. Please provide valid doubles.", file=sys.stderr)
    #     sys.exit(1)

    # SpaceCirclefor4(P1, P2, P3, P4)
    # #print("Generate success.")
    # sys.exit(0)
    #圆心插值
    # P1 = [0,0,0,180,-20,0]
    # P2 = [0,0,0,180,-20,180]
    # rxyz = Pose_slerp(P1,P2,181)
    # print(rxyz[0][0])
    # P_circle = [350.707101,281.290867,449.754,180,-20,0]
    # n = 360
    # z = 30.647245  #物体高度
    # x = 56.68305607
    # p_xyz,p_joint,P = getCircleTraj(P_circle,n,z,x)
    # # print(p_xyz)
    # # show the result
    # p = np.array(p_xyz)
    # fig=plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot3D(p[:,0],p[:,1],p[:,2])
    # ax.set_zlabel('Z')
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')
    # ax.scatter3D(P[0],P[1],P[2])
    # ax.scatter3D(p[0][0],p[0][1],p[0][2])
    # ax.scatter3D(p[89][0],p[89][1],p[89][2])
    # ax.scatter3D(p[179][0],p[179][1],p[179][2])
    # ax.scatter3D(p[269][0],p[269][1],p[269][2])
    # plt.show()
    #半球采图
    p_xyz,p_joint = getTcppoint(rt_tcp=RT_tcp_sort)
    p = np.array(p_xyz)
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.set_xlim([0,1000])
    ax.set_ylim([-300,700])
    ax.set_zlim([0,1000])
    ax.scatter3D(p[:,0],p[:,1],p[:,2],'.')
    ax.scatter3D(0,0,0,c='r',marker='o',linewidths = 10)
    # for i in range(len(p)):
    #     ax.scatter3D(p[i,0],p[i,1],p[i,2])
    plt.show()

    # 四个点
    # p1 = [360.193,259.177,769.122,178.919,-12.425,0.565]
    # p2 = [360.193+100,259.177+100,769.122,174.191,-9.986,1.586]
    # p3 = [360.193+200,259.177,769.122,166.356,-3.351,2.255]
    # p4 = [360.193+100,259.177-100,769.122,165.236,-5.254,3.415]
    # p1 = np.array([139.119,278.196,511.319,177.998,-13.347,0])
    # p2 = np.array([353.407,54.005,511.319,177.995,-13,90])
    # p3 = np.array([566.991,223.053,511.319,175.857,-13,180])
    # p4 = np.array([371.640,488.093,511.319,177.995,-13,-90])
    # p1 = np.array([148.665,285.584,317.639,178.673,-20,0.003])
    # p2 = np.array([347.194,63.796,317.639,179.374,-20,89.981])
    # p3 = np.array([563.034,265.257,317.639,178.001,-20,179.999])
    # p4 = np.array([356.477,487.213,317.639,178.991,-20.002,-90.001])
    # p,Rc,Rw = SpaceCirclefor4(p1,p2,p3,p4,45)
    
    # show the result
    # p = np.array(p)
    # fig=plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot3D(p[:,0],p[:,1],p[:,2])
    # ax.set_zlabel('Z')
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')
    # ax.scatter3D(p1[0],p1[1],p1[2])
    # ax.scatter3D(p2[0], p2[1], p2[2])
    # ax.scatter3D(p3[0], p3[1], p3[2])
    # ax.scatter3D(p4[0], p4[1], p4[2])
    # ax.scatter3D(Rc[0], Rc[1], Rc[2])
    # ax.scatter3D(Rw[0], Rw[1], Rw[2])
    # plt.show()

    #三个点
    # p1 = [360.193,259.177,769.122,178.919,-12.425,0.565]
    # p2 = [360.193+100*(1-np.sin(np.pi/4)),259.177+100*np.sin(np.pi/4),769.122,174.191,-9.986,1.586]
    # p3 = [360.193+100,259.177+100,769.122,166.356,-3.351,2.255]
    # p1 = [360.193,0,769.122,178.919,-12.425,0.565]
    # p2 = [360.193+100*(1-np.sin(np.pi/4)),0+100*np.sin(np.pi/4),769.122,174.191,-9.986,1.586]
    # p3 = [360.193+100,0+100,769.122,166.356,-3.351,2.255]
    # p,Rc = SpaceCirclefor3(p1,p2,p3,45)
    # print(f'p1:{p1}--p2{p2}--p3{p3}')
    # show the result
    # p = np.array(p)
    # fig=plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot3D(p[:,0],p[:,1],p[:,2])
    # ax.set_zlabel('Z')
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')
    # ax.scatter3D(p1[0],p1[1],p1[2])
    # ax.scatter3D(p2[0], p2[1], p2[2])
    # ax.scatter3D(p3[0], p3[1], p3[2])
    # ax.scatter3D(Rc[0], Rc[1], Rc[2])
    # plt.show()

    #单点
    # P = [360.193,259.177,769.122,178.919,-12.425,0.565]
    # m = RandData(P,200,10,5)
    # #show the result
    # m = np.array(m)
    # fig=plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_zlabel('Z')
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')
    # for i in range(200):
    #     ax.scatter3D(m[i,0],m[i,1],m[i,2])
    # plt.show()
