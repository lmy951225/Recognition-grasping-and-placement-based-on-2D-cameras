import os
import sys

import cv2
import open3d as o3d
from typing import List
import math

sys.path.append(os.getcwd())
base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
sys.path.append('AgilebotController')

import grpc
import points_pb2 as pb2
import points_pb2_grpc as pb2_grpc
import csv
from AgilebotController.grpc_module.client_robot import RobotClient
# from MotionPlanning.grpc_module.client_motionpalnning import MotionPlanningClient
from RuckigP2P import *
from sisiphy_grpc.point import Point
from TritionModule.test_del import infer_client
from driver.grasp.gripper3 import gripper_client
from photo import Photo
import numpy as np
from place import *
import threading
import time
# from Pose.getpose import RandData

#全局变量
#机械臂
robotClient = RobotClient()
robotClient.test_connect()
robotClient.test_setSpeedRatio(0.2)
#夹爪
lmy = gripper_client(port="/dev/ttyUSB0")
POINT = Point()
#相机参数
camera=cv2.VideoCapture(-1)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
print("相机曝光值为{}ms".format(camera.get(15)))
#指定参数
tcp = np.array([299.921,267.846,550,180,0,0])
flag1 = "1_0_y_m"
flag2 = "2_0_y_m"
template_tcp = np.array([502.223,-320.919,750,180,0,0])
center_point=np.array([666.786, 351.550, 296.557, 180, 0, 0 ])
home_point = np.array([571.261, 0.917, 521.650, 180, 0, 0 ])
grab_point = np.array([571.261, 0.917, 471.650, 180, 0, 0 ])
place_pose_tcp = np.array([571.261, 0.917, 536.650, 180, 0, 0])
tray_x_d = 295
tray_y_d = 295
part_x_d = 80
part_y_d = 75
scale = 522/147.5
flag_chose = 'ys'
count_limit = 3
position=5
current=0.19
idx_internel = 0
T_zhiju = np.array([[   0.99982834  ,  0.01824984  , -0.00320016 ,  74.0064927 ],
                        [  -0.01829169  ,  0.99974051 ,  -0.01357643,  343.84237348],
                        [   0.00295156   , 0.01363264   , 0.99990272, -224.92352786],
                        [   0.         ,   0.        ,    0.        ,    1.        ]])

class Pt():
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

class StagePose():
    def __init__(self,stage_angle,stage_position):
        self.stage_angle = stage_angle
        self.stage_positon = stage_position


class GraspPlanningClient():
    def __init__(self):
        self.conn = grpc.insecure_channel("0.0.0.0:49996")
        self.client = pb2_grpc.GraspPlanningStub(channel=self.conn)
        # print(f'self.client:{self.client}')

    def test_get2Points(self):
        p1 = pb2.Pt(x=571.610046, y=10.440644, z = 222.961029)
        p2 = pb2.Pt(x=571.565735 ,y =-10.549973,z = 219.105759)
        req = pb2.PointInput(P1=p1,P2=p2)
        self.client.get2Points(req)
        reply = self.client.get2Points(req)
        print(reply.tcpPoint.x)

    def test_hello(self):
        req = pb2.HelloGrpcReq(name='Diamon', age=27)
        self.client.HelloGrpc(req)
    
    def client_get2Points(self, Pt1, Pt2):
        p1 = pb2.Pt(x=Pt1.x, y=Pt1.y, z = Pt1.z)
        p2 = pb2.Pt(x=Pt2.x, y=Pt2.y, z = Pt2.z)
        req = pb2.PointInput(P1=p1,P2=p2)
        self.client.get2Points(req)
        reply = self.client.get2Points(req)
        return reply.tcpPoint.x, reply.tcpPoint.y, reply.tcpPoint.z, reply.tcpPoint.rx, reply.tcpPoint.ry, reply.tcpPoint.rz

    def client_get3Points(self, Pt1, Pt2,Pt3):
        p1 = pb2.Pt(x=Pt1.x, y=Pt1.y, z = Pt1.z)
        p2 = pb2.Pt(x=Pt2.x, y=Pt2.y, z = Pt2.z)
        p3 = pb2.Pt(x=Pt3.x, y=Pt3.y, z = Pt3.z)
        req = pb2.PointInput(P1=p1,P2=p2,P3=p3)
        self.client.get3Points(req)
        reply = self.client.get3Points(req)
        return reply.tcpPoint.x, reply.tcpPoint.y, reply.tcpPoint.z, reply.tcpPoint.rx, reply.tcpPoint.ry, reply.tcpPoint.rz

    def client_get2PointsMotionPlanning(self, Pt1, Pt2, object_mesh_path, init_points,angle,position):
        p1 = pb2.Pt(x=Pt1.x, y=Pt1.y, z = Pt1.z)
        p2 = pb2.Pt(x=Pt2.x, y=Pt2.y, z = Pt2.z)
        stage = pb2.StagePose(stage_angle=angle,stage_position=position)
        init_points = pb2.JointPosition(J1=init_points[0],J2=init_points[1],J3=init_points[2],J4=init_points[3],J5=init_points[4],J6 =init_points[5])
        req = pb2.PointInput(P1=p1,P2=p2,object_mesh_path=object_mesh_path, init_points=init_points,stage_pose=stage)
        print("=========0==========")
        reply = self.client.get2PointsMotionPlanning(req)
        print("=========1==========")
        planning_points = reply.points
        # tcp_point = reply.tcp_point
        joint_points = list()

        for item in planning_points:
            joint_points.append([item.J1,item.J2,item.J3,item.J4,item.J5,item.J6]) 
        print(joint_points)
        return reply.tcpPoint,joint_points

def shot_one(i):
    # camera=cv2.VideoCapture(-1)
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    # # camera.set(cv2.CAP_PROP_FPS, 30)
    # camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    if i == 1:
        for j in range(50):
            camera.grab()  
    else:
        for j in range(4):
            camera.grab()
    # for j in range(50):
    #     camera.grab()
    (grabbed, img) = camera.read()
    firename=str('./test_img_'+str(i)+'.jpg')
    cv2.imwrite(firename, img)
    # camera.release()
    return img,firename

def photo_one(i):
    hk = Photo()
    img,firename = hk.cam(i)
    return img,firename


def get_bbx_points(path="Product_FXH_001.stl"):
    pcd = o3d.io.read_triangle_mesh(path)
    aabb = pcd.get_axis_aligned_bounding_box()
    ab = aabb.get_center()
    x = aabb.get_extent()
    P1 = ab - 0.5*x
    P2 = ab + 0.5*x
    P_min = np.array([P1[0],P1[1],P1[2]])
    P_max = np.array([P2[0],P2[1],P2[2]])
    P = np.array([P_min,P_max])
    print(P)
    return P 

def rukig_p2p(P0,P1,i):
    P0_joint = c.transferCartesian2Joints(P0)
    P1_joint = c.transferCartesian2Joints(P1)
    # print(P0_joint,P1_joint)
    thetas_list = np.array([[*P0_joint],
                            [*P1_joint]])
    curent_vel = np.array([0,0,0,0,0,0])
    target_vel = np.array([0,0,0,0,0,0])
    curent_acc = np.array([0,0,0,0,0,0])
    target_acc = np.array([0,0,0,0,0,0])
    T_s = 0.001
    q_boundary = np.array([[-2.94, -2.33, -1.2, -3.29, -1.98, -6.26], 
                           [ 2.94,  1.72,  3.47,  3.29,  1.98,  6.26]])
    v_boundary = np.array([[-5.71, -4.56, -5.71, -7.75, -6.97, -10.46],
                          [5.71, 4.56, 5.71, 7.75, 6.97, 10.46]])
    
    a_boundary = np.array([[-23, -19, -23, -31, -27, -39],
                          [23, 19, 23, 31, 27, 39]])
    j_boundary = np.array([[-230, -190, -230, -310, -270, -390],
                          [230, 190, 230, 310, 270, 390]])
    
    T, all_Q, all_V, all_A, all_J, duration,inter_time = multi_dimensional_p2p_ruckig(thetas_list[0],thetas_list[1],curent_vel,target_vel,curent_acc,target_acc,v_boundary,a_boundary,j_boundary,T_s)
    detailed_trajecory = csv_traj(T,all_Q,all_V,all_A,all_J)
    filename = 'traj' + str(i)
    output = detailed_trajecory.to_csv('AgilebotController/home/host_dir/robot_controller_tmp/trajectory/' + filename + '.csv',index=False)
    # robotClient.test_executeFlyShotTraj('traj')
    return filename

def GraspState(positon,current):
        while True:
            print("当前位置{},当前电流{}".format(lmy.get_position(),lmy.get_current()))
            if np.abs(lmy.get_current()) > current:
                print("当前位置{},当前电流{}".format(lmy.get_position(),lmy.get_current()))
                if lmy.get_position() > positon:
                    print("抓取成功")
                    return True
                else:
                    print("抓取失败")
                    return False
            else:
                print("当前位置{},当前电流{}".format(lmy.get_position(),lmy.get_current()))
                if lmy.get_position() < positon:
                    print("抓取失败")
                    return False

def FallState():
    while True:
        num = lmy.get_State()
        if num == 0:
            print("到位")
        if num == 1:
            print("运动中")
        if num == 2:
            print("夹持")
            return True
        if num == 3:
            print("掉落") 
            return False      
        
def capture_for_yolo(tcp=np.array([299.921,267.846,550,180,0,0]),flag="1_0_y_m"):
    """
    Args:
        tcp : 初始位姿, np.array([299.921,267.846,550,180,0,0]).
        flag : 1号位置 x/y/n 分别代表x/y轴/就近位置 m代表多工件

    Returns:
        cameraTcppose0: 多个工件的位姿,list
    """
    robotClient.test_MovePose(*tcp)
    x0,y0,z0,rx0,ry0,rz0 = robotClient.test_getPose()
    _,firename = shot_one(1)
    transformation0 = POINT.xyz_rxryrz2transformation(np.array([x0,y0,z0,rx0,ry0,rz0]))
    cameraTcppose0, _ = infer_client(img_path = firename, box = np.array([643,407,float(643+392),float(407+392)]),transform = transformation0,flag=flag)
    return cameraTcppose0

def capture_for_grasp(cameraTcppose0,flag="2_0_y_m"):
    """
    Args:
        cameraTcppose0 : 1号yolo识别出工件后应到的拍照姿态,list
        flag : 2号位置 x/y/n 分别代表x/y轴/就近位置 m代表多工件
    """
    tcp0 = np.copy(cameraTcppose0)
    camera_pose_tcp0 = POINT.transformation2xyz_rxryrz(tcp0)
    print(f'camera_pose_tcp1================{camera_pose_tcp0}')
    robotClient.test_MovePose(camera_pose_tcp0[0],camera_pose_tcp0[1],camera_pose_tcp0[2],camera_pose_tcp0[3],camera_pose_tcp0[4],camera_pose_tcp0[5])
    x2,y2,z2,rx2,ry2,rz2 = robotClient.test_getPose()
    _,firename = shot_one(2)
    transformation2 = POINT.xyz_rxryrz2transformation(np.array([x2,y2,z2,rx2,ry2,rz2]))
    cameraTcppose2,pose2 = infer_client(img_path = firename, box = np.array([0,0,float(0),float(0)]),transform = transformation2,flag=flag)
    pose_world = np.copy(pose2)
    tcp_byd = POINT.transformation2xyz_rxryrz(pose_world)
    print(f'pose_obj================{tcp_byd}')
    tcp2 = np.copy(cameraTcppose2)
    camera_pose_tcp2 = POINT.transformation2xyz_rxryrz(tcp2)
    print(f'camera_pose_tcp2================{camera_pose_tcp2}')

    # condition for byd
    assert pose2[2,3] > 18
    assert pose2[2,3] < 36
    # assert tcp2[2,3] > 255
    assert tcp2[2,3] > 265

    print("x,y,z",camera_pose_tcp2[0]-50*tcp2[0,2],
                camera_pose_tcp2[1]-50*tcp2[1,2],
                camera_pose_tcp2[2]-50*tcp2[2,2])

    #移动到抓取点上方50mm
    robotClient.test_MovePose(camera_pose_tcp2[0]-50*tcp2[0,2],
                camera_pose_tcp2[1]-50*tcp2[1,2],
                camera_pose_tcp2[2]-50*tcp2[2,2],
                camera_pose_tcp2[3],camera_pose_tcp2[4],camera_pose_tcp2[5])

    #move_line 到抓取点
    robotClient.test_MovePose(camera_pose_tcp2[0]-0*tcp2[0,2],
                camera_pose_tcp2[1]-0*tcp2[1,2],
                camera_pose_tcp2[2]-0*tcp2[2,2],
                camera_pose_tcp2[3],camera_pose_tcp2[4],camera_pose_tcp2[5])   
    
def better_grasp(position=5,current=0.19):
    lmy.open(0)
    ret = GraspState(position,current)
    return ret      
     
def process_for_adjust(template_tcp = np.array([502.223,-320.919,750,180,0,0]),num_iter=2):
    robotClient.test_MovePose(*template_tcp)
    for idx in range(num_iter):
        print(f'第{idx+1}次矫正\n')
        x3,y3,z3,rx3,ry3,rz3 = robotClient.test_getPose()
        transformation3 = POINT.xyz_rxryrz2transformation(np.array([x3,y3,z3,rx3,ry3,rz3]))
        img,firename = photo_one(3)
        flag="3_0_y_s"
        cameraTcppose3,_ = infer_client(img_path = firename, box = np.array([643,407,float(643+392),float(407+392)]),transform = transformation3,flag=flag)
        print(f"tcp3_0 ================ {cameraTcppose3}")
        tcp3_0 = np.copy(cameraTcppose3)
        T_corection = POINT.transformation2xyz_rxryrz(tcp3_0)
        print(f'T_corection:{T_corection}')
        robotClient.test_MovePose(*T_corection)
    T_final = T_zhiju @ tcp3_0
    T_final = POINT.transformation2xyz_rxryrz(T_final)
    print(f'T_final:{T_final}')
    robotClient.test_MovePose(T_final[0],T_final[1],T_final[2],T_final[3],T_final[4],T_final[5])
    robotClient.test_setSpeedRatio(0.1)
    time.sleep(0.1)
    robotClient.test_MovePose(T_final[0],T_final[1],T_final[2]-64,T_final[3],T_final[4],T_final[5])
    lmy.open(10)
    robotClient.test_BiasMovePose(bias_z=100)
    
def process_no_adjust(place_pose_tcp = np.array([571.261, 0.917, 536.650, 180, 0, 0])):
    robotClient.test_MovePose(*place_pose_tcp)
    robotClient.test_MovePose(place_pose_tcp[0],place_pose_tcp[1],place_pose_tcp[2]-71,place_pose_tcp[3],place_pose_tcp[4],place_pose_tcp[5])
    lmy.open(10)
    robotClient.test_MovePose(place_pose_tcp[0],place_pose_tcp[1],place_pose_tcp[2]+100,place_pose_tcp[3],place_pose_tcp[4],place_pose_tcp[5])

def process_for_place(center_point=np.array([666.786, 351.550, 303, 180, 0, 0 ]),
                      tray_x_d=295,tray_y_d=295,part_x_d=80,part_y_d=75,scale=522/147.5,flag_chose='ys'):
    
    x_p,y_p,z_p,rx_p,ry_p,rz_p = robotClient.test_getPose()
    tcp,nx,ny,cam = get_Place_tcp(center_point,part_x_d,part_y_d,tray_x_d,tray_y_d,1920,1080,scale)
    tcp_pose,_ = tcp_chose(tcp,cam,nx,ny,flag_chose)
    global idx_internel
    for pose in [tcp_pose[idx_internel]]:
        robotClient.test_MovePose(pose[0],pose[1],z_p,pose[3],pose[4],pose[5])
        robotClient.test_MovePose(*pose)
        lmy.open(10)
        robotClient.test_MovePose(pose[0],pose[1],z_p,pose[3],pose[4],pose[5])
        idx_internel += 1
        
def process_for_place_from_point(center_point=np.array([666.786, 351.550, 303, 180, 0, 0 ]),home_point = np.array([571.261, 0.917, 521.650, 180, 0, 0 ]),
                                grab_point = np.array([571.261, 0.917, 471.650, 180, 0, 0 ]),tray_x_d=295,tray_y_d=295,
                                part_x_d=80,part_y_d=80,scale=522/147.5,flag_chose='ys',idx_loop=0):
    tcp,nx,ny,cam = get_Place_tcp(center_point,part_x_d,part_y_d,tray_x_d,tray_y_d,1920,1080,scale)
    tcp_pose,_ = tcp_chose(tcp,cam,nx,ny,flag_chose)
    global idx_internel
    for pose in [tcp_pose[idx_internel]]:
        robotClient.test_MovePose(*home_point)
        robotClient.test_MovePose(*grab_point)
        lmy.open(0)
        robotClient.test_MovePose(*home_point)
        robotClient.test_MovePose(pose[0],pose[1],home_point[2],pose[3],pose[4],pose[5])
        robotClient.test_MovePose(*pose)
        lmy.open(10)
        robotClient.test_MovePose(pose[0],pose[1],home_point[2],pose[3],pose[4],pose[5])
        idx_internel += 1
        
def main(count_limit):
    count = 0
    while count < count_limit: 
        failGrasp = False
        cameraTcppose0 = capture_for_yolo(tcp,flag1)
        #2号
        for idx_loop,value in enumerate(cameraTcppose0):
            capture_for_grasp(value,flag2)
            #是否抓取成功
            ret = better_grasp(position,current)
            #放置前是否掉落
            if ret:
                robotClient.test_BiasMovePose(bias_z=200)
                ret2 = FallState()
                #放置
                if ret2:
                    process_for_place(center_point,tray_x_d,tray_y_d,part_x_d,part_y_d,scale,flag_chose)
                else:
                    lmy.close()
                    raise ValueError("物体掉落")
            else:
                failGrasp = True
                lmy.open(10)
        count += 1 
        if not failGrasp:
            break
    lmy.close()
    robotClient.test_MovePose(*tcp)
    

if __name__ == '__main__':
    main(count_limit)
    
