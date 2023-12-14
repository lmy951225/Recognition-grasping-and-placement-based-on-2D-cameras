import cv2
import numpy as np
import glob
from math import *
import pandas as pd
import os
import sys
import time
sys.path.append("..")
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

#建立存放寻找棋盘格角点照片及位姿文件的文件夹
time_now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
folder = os.getcwd()+'/findcheseboard'+'_'+time_now
os.mkdir(folder)

#图像处理
def img_process(filename,flag=2):
    '''
    param:filename:图片路径
    parame:i:翻转,0:垂直,1:水平,-1:水平+垂直
    '''
    idx=1
    filelist=os.listdir(filename)
    if flag == 0 or flag == -1 or flag == 1:
        folder_img = filename +'_'+time_now
        os.mkdir(folder_img)
        for img_file in filelist:
            img=cv2.imread(filename+'/'+img_file)
            img=cv2.flip(img,flag)
            cv2.imwrite(folder_img + '/' + img_file,img)
            print(idx)
            idx+=1
        filename = folder_img
    else:
        filename = filename
    return filename

#获取tcp姿态
def str2num(pose_str: str):

    if isinstance(pose_str, str):

        if '{' in pose_str:
            pose_str = pose_str.replace('{', '')
        if '}' in pose_str:
            pose_str = pose_str.replace('}', '')
        if '*PI/180' in pose_str:
            pose_str = pose_str.replace('*PI/180', '')

    return float(pose_str)

def getPose(file_name = 'pose.txt', dof=6, columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']):
    pre_df = pd.read_csv(file_name, header=None)
    df = pre_df.iloc[:,:dof].copy()

    for idx,row in pre_df.iloc[:,:dof].iterrows():
        for i,p in enumerate(row):
            # print(idx,i,p, str2num(p))
            df.loc[idx,i] = str2num(p)

    df.columns = columns
    # print(df)
    # print(df.describe())
    return df

#计算内参
def Intrinsic_calib(chess_board_x_num,chess_board_y_num,chess_board_len,filename):
    # 找棋盘格角点
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值
    #棋盘格模板规格
    w = chess_board_x_num   # 12 - 1
    h = chess_board_y_num   # 9  - 1
    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    # objp = objp*15  # 15 mm
    objp = objp*chess_board_len  # 5 mm

    # 储存棋盘格角点的世界坐标和图像坐标s对
    objpoints = [] # 在世界坐标系中的三维点
    imgpoints = [] # 在图像平面的二维点
    #加载pic文件夹下所有的jpg图像
    images = glob.glob(filename+'/*.jpg')  #   拍摄的十几张棋盘图片所在目录
    i = 1
    for fname in images[0:]:
        # print(fname)
        img = cv2.imread(fname)
        # 获取画面中心点
        #获取图像的长宽
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # 找到棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
        # print(f'corner.shape{corners.shape}')
        # print(f'objp.shape{objp.shape}')
        # 如果找到足够点对，将其存储起来
        if ret == True:
            print(f"第{i}张找到角点的图片:", fname)
            i = i+1
            # 在原角点的基础上寻找亚像素角点
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            #追加进入世界三维点和平面二维点中
            objpoints.append(objp)
            imgpoints.append(corners)
            # print(corners)
            # 将角点在图像上显示
            cv2.drawChessboardCorners(img, (w,h), corners, ret)
            nn = folder + '/' + fname[-10:-4]+'.jpg'
            cv2.imwrite(nn, img)
    # #标定
    # print('正在计算')
    ret, mtx, dist, rvecs, tvecs = \
        cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # print("ret:",ret  )
    # print("mtx:\n",mtx)      # 内参数矩阵
    # print("dist畸变值:\n",dist   )   # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    return mtx,dist

#用于根据欧拉角计算旋转矩阵
def myRPY2R_robot(x, y, z):
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz@Ry@Rx
    return R

#用于根据位姿计算变换矩阵
def pose_robot(x, y, z, Rx, Ry, Rz, cali_m = 'eye_in_hand'):
    thetaX = Rx / 180 * pi
    thetaY = Ry / 180 * pi
    thetaZ = Rz / 180 * pi
    R = myRPY2R_robot(thetaX, thetaY, thetaZ)

    t = np.array([[x], [y], [z]])
    RT1 = np.column_stack([R, t])  # 列合并
    RT1 = np.row_stack((RT1, np.array([0,0,0,1])))
    if cali_m =='eye_to_hand':
        RT1=np.linalg.inv(RT1)
    return RT1

#用来从棋盘格图片得到相机外参
def get_RT_from_chessboard(img_path,chess_board_x_num,chess_board_y_num,K,chess_board_len,dist_coef):
    '''
    :param img_path: 读取图片路径
    :param chess_board_x_num: 棋盘格x方向格子数
    :param chess_board_y_num: 棋盘格y方向格子数
    :param K: 相机内参
    :param chess_board_len: 单位棋盘格长度,mm
    :return: 相机外参
    '''
    print('img_path:', img_path)
    img=cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (chess_board_x_num, chess_board_y_num), None)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值
    print('ret is: {}'.format(ret))
    if ret:
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # print('corners shape is: {}, data: {}'.format(corners.shape, corners))
        # print('='*100)
        corner_points=np.zeros((2,corners.shape[0]),dtype=np.float32)
        for i in range(corners.shape[0]):
            corner_points[:,i]=corners[i,0,:]
        object_points=np.zeros((chess_board_x_num*chess_board_y_num,3),dtype=np.float32)
        object_points[:,:2] = np.mgrid[0:chess_board_x_num,0:chess_board_y_num].T.reshape(-1,2)
        object_points *= chess_board_len
        
        retval,rvec,tvec  = cv2.solvePnP(object_points,corner_points.T, K, distCoeffs=dist_coef)
        # print('*'*100)
        # print(rvec, tvec)
        # print(f'object_points:{object_points.T}')
        
        
        imgpoints2, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coef)
        # print('ss',imgpoints2.shape)
        error = cv2.norm(corner_points.T, imgpoints2.squeeze(), cv2.NORM_L2)
        
        

        RT=np.column_stack(((cv2.Rodrigues(rvec))[0],tvec))
        RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
        # print("相机外参矩阵:\n",RT)
        
    else:
        RT = np.eye(4)
    return RT,error,object_points,corner_points.T,size

def main(filename_jpg,filename_pose,cali_m,chess_board_x_num,chess_board_y_num,chess_board_len,flag):
    '''
    param:filename_jpg:原始图片文件夹名称 'byd_0918_11'
    param:filename_pose:tcp位姿文件 'tcp_0918_11.txt'
    param:cali_m:'eye_to_hand眼在手外 'eye_in_hand'眼在手上
    param:chess_board_x_num #棋盘格x方向格子数 12
    param:chess_board_y_num #棋盘格y方向格子数 9
    param:chess_board_len #单位棋盘格长度,mm  5
    param:flag:图片处理,-1(中心对称),0(垂直),1(水平),2(不翻转)
    '''
    filename_jpg = filename_jpg
    chess_board_x_num=chess_board_x_num - 1 #棋盘格x方向格子数
    chess_board_y_num=chess_board_y_num - 1 #棋盘格y方向格子数
    chess_board_len=chess_board_len #单位棋盘格长度,mm
    new_filename = img_process(filename_jpg,flag)
    K,dist_coef = Intrinsic_calib(chess_board_x_num,chess_board_y_num,chess_board_len,new_filename)
    df = getPose(file_name = filename_pose, dof=6)

    R_all_end_to_base = []
    T_all_end_to_base = []
    R_all_chess_to_cam = []
    T_all_chess_to_cam = []

    #计算board to cam 变换矩阵
    R_all_chess_to_cam_1=[]
    T_all_chess_to_cam_1=[]
    total_err = 0
    objs = []
    corners = []
    # RT_list = []
    for i in range(len(df)):
        image_path = os.path.join(new_filename, 'img-'+str(i+1)+'.jpg')
        RT,err,obj,corner,size=get_RT_from_chessboard(image_path, chess_board_x_num, chess_board_y_num, K, chess_board_len, dist_coef=dist_coef)
        print(f'第{i+1}个图片平均误差为:{err}')
        print('*'*100)
        total_err += err*err
        # RT_list.append(RT)
        # print(f'corner.shape{corner[:,None,:].shape}')
        # print(f'obj.shape{obj.shape}')
        # print('obj_type',obj.dtype,'corners_type',corner.dtype)
        objs.append(obj)
        corners.append(corner[:,None,:])
        R_all_chess_to_cam_1.append(RT[:3,:3])
        T_all_chess_to_cam_1.append(RT[:3, 3].reshape((3,1)))

    total_err /= (len(df) * chess_board_x_num * chess_board_y_num)
    rms_err = sqrt(total_err)
    print(f"所有角点的重投影误差:{rms_err}")
    # ret, mtx, dis, rvecs, tvecs = \
    #     cv2.calibrateCamera(objs, corners, size, None, None)
    # print(f'ret:{ret}')
    # print(f'mtx:{mtx}')
    # print(f'dis:{dis}')
    # np.save('rt_list.npy',RT_list)
    #计算end to base变换矩阵
    R_all_end_to_base_1=[]
    T_all_end_to_base_1=[]

    for idx,row in df.iterrows():
        RT=pose_robot(row[0], row[1], row[2], row[3], row[4], row[5],cali_m)
        R_all_end_to_base_1.append(RT[:3, :3])
        T_all_end_to_base_1.append(RT[:3, 3].reshape((3, 1)))

    for i in range(len(df)):
        R_all_end_to_base.append(R_all_end_to_base_1[i])
        T_all_end_to_base.append(T_all_end_to_base_1[i])
        R_all_chess_to_cam.append(R_all_chess_to_cam_1[i])
        T_all_chess_to_cam.append(T_all_chess_to_cam_1[i])

    R,T=cv2.calibrateHandEye(R_all_end_to_base,T_all_end_to_base,
                                R_all_chess_to_cam,T_all_chess_to_cam, 
                             method=0) #手眼标定

    RT=np.column_stack((R,T))
    RT = np.row_stack((RT, np.array([0, 0, 0, 1]))) #即为cam to end变换矩阵
    # print('+'*100)
    # print('Tsai法相机相对于末端的变换矩阵为:')
    # print(RT)
    # 结果验证
    print('='*100)
    for i in range(len(df)):

        RT_end_to_base=np.column_stack((R_all_end_to_base_1[i],T_all_end_to_base_1[i]))
        RT_end_to_base=np.row_stack((RT_end_to_base,np.array([0,0,0,1])))
        # print(RT_end_to_base)
        data1=pd.DataFrame(RT_end_to_base)
        data1.to_csv(folder + '/RT_end_to_base.txt',sep=' ',index=0,header=0,mode='a')

        RT_chess_to_cam=np.column_stack((R_all_chess_to_cam_1[i],T_all_chess_to_cam_1[i]))
        RT_chess_to_cam=np.row_stack((RT_chess_to_cam,np.array([0,0,0,1])))
        # print(RT_chess_to_cam)
        data2=pd.DataFrame(RT_chess_to_cam)
        data2.to_csv(folder + '/RT_chess_to_cam.txt',sep=' ',index=0,header=0,mode='a')
        

        RT_cam_to_end=np.column_stack((R,T))
        RT_cam_to_end=np.row_stack((RT_cam_to_end,np.array([0,0,0,1])))
        # print(RT_cam_to_end)
        data3=pd.DataFrame(RT_cam_to_end)
        data3.to_csv(folder + '/RT_cam_to_end.txt',sep=' ',index=0,header=0,mode='a')

        # p = np.array([[0],[0],[0],[1]])
        # RT_chess_to_base=RT_end_to_base@RT_cam_to_end@RT_chess_to_cam@p#即为选取的点在机器人基坐标系下的位置
        # print(f'第{i+1}次结果为:{RT_chess_to_base.T}')

    return K,dist_coef,RT

if __name__=='__main__':
    #眼在手上,图片不预处理
    filename_jpg = 'cam_end_3_2_flip_16' #棋盘格图片文件夹
    filename_pose = 'cam_end_3_2_flip_16.txt' #tcp位姿文件
    cali_m = 'eye_in_hand' #眼在手上的方式
    chess_board_x_num = 12 #棋盘格x方向格子数 
    chess_board_y_num = 9 #棋盘格y方向格子数 
    chess_board_len = 15 #棋盘格一个方格尺寸
    flag = 2 #棋盘格图片是否处理的标志(1:水平翻转，0：垂直翻转，-1：中心对称，2：不处理)
    K1,dit1,RT = main('cam_end_3_2_flip_16','cam_end_3_2_flip_16.txt','eye_in_hand',12,9,15,2)
    print(f'相机内参:\n{K1}')
    print(f'相机畸变:\n{dit1}')
    print(f'手眼矩阵(eye in hand):\n{RT}')

    #眼在手上,图片中心对称
    # filename_jpg = 'cam_end_3_2_1'
    # filename_pose = 'cam_end_3_2_1.txt'
    # cali_m = 'eye_in_hand'
    # chess_board_x_num = 12
    # chess_board_y_num = 9
    # chess_board_len = 15
    # flag = -1
    # K1,dit1,RT = main(filename_jpg,filename_pose,cali_m,chess_board_x_num,chess_board_y_num,chess_board_len,flag)
    # print(f'相机内参:\n{K1}')
    # print(f'相机畸变:\n{dit1}')
    # print(f'手眼矩阵(eye in hand):\n{RT}')

    #眼在手外,图片不预处理
    # filename_jpg = 'byd_0918_11'
    # filename_pose = 'tcp_0918_11.txt'
    # cali_m = 'eye_to_hand'
    # chess_board_x_num = 12
    # chess_board_y_num = 9
    # chess_board_len = 5
    # flag = 2
    # K2,dit2,RT2 = main('byd_0918_11','tcp_0918_11.txt','eye_to_hand',12,9,5,2)
    # print(f'相机内参:\n{K2}')
    # print(f'相机畸变:\n{dit2}')
    # print(f'手眼矩阵(eye to hand):\n{RT2}')
    