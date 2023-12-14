import numpy as np
import pandas as pd
import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

def loadtxtmethod(filename):
    data = np.loadtxt(filename,dtype=np.float32,delimiter=' ')
    return data

def loadtxtmethod2(filename):
    data = np.loadtxt(filename,dtype=np.float32,delimiter=',')
    return data

def error_compute(filename_RT_chess_to_cam,filename_RT_cam_to_end,filename_RT_end_to_base,filename_real_base,filename_real_cam):
    '''
    param:filename_RT_chess_to_cam:相机外参矩阵文件
    param:filename_RT_cam_to_end:手眼转换矩阵文件
    param:filename_RT_end_to_base:拍照时tcp位姿矩阵文件
    param:filename_real_base:选取点在机械臂坐标下的位置文件or在tcp下的位置文件
    param:filename_real_cam:选取点在相机坐标下的位置文件
    return : 每个点与真实值的误差以及误差的均值
    '''
    RT_cam_to_end = loadtxtmethod(filename_RT_cam_to_end)
    RT_end_to_base = loadtxtmethod(filename_RT_end_to_base)
    RT_chess_to_cam = loadtxtmethod(filename_RT_chess_to_cam)
    real_base = loadtxtmethod2(filename_real_base)
    real_cam = loadtxtmethod2(filename_real_cam)
    err_single = []
    err = 0
    err_all = 0
    # print(np.size(real_base[0]))
    if np.size(real_base[0]) == 1:
        for idx in range(int(RT_chess_to_cam.shape[0]/4)):
            RT_chess_to_base = RT_end_to_base[idx*4:(idx+1)*4,:]@RT_cam_to_end[idx*4:(idx+1)*4,:]@RT_chess_to_cam[idx*4:(idx+1)*4,:]@real_cam.T
            # print(f'第1个点在第{idx+1}张图片计算值为:{RT_chess_to_base.T}')
            err += np.sqrt((RT_chess_to_base[0] - real_base[0])**2 + (RT_chess_to_base[1] - real_base[1])**2 +  (RT_chess_to_base[2] - real_base[2])**2)
            # print(f'第1个点在第{idx+1}张图片误差为:{err}')
        err_single.append(err / RT_chess_to_cam.shape[0] * 4) 
    else:
        for i in range(real_base.shape[0]):
            for idx in range(int(RT_chess_to_cam.shape[0]/4)):
                RT_chess_to_base = RT_end_to_base[idx*4:(idx+1)*4,:]@RT_cam_to_end[idx*4:(idx+1)*4,:]@RT_chess_to_cam[idx*4:(idx+1)*4,:]@real_cam[i,:].T
                # print(f'第{i+1}个点在第{idx+1}张图片计算值为:{RT_chess_to_base.T}')
                err += np.sqrt((RT_chess_to_base[0] - real_base[i][0])**2 + (RT_chess_to_base[1] - real_base[i][1])**2 +  (RT_chess_to_base[2] - real_base[i][2])**2)
                # print(f'第{i+1}个点在第{idx+1}张图片误差为:{err}')
            err_single.append(err / RT_chess_to_cam.shape[0] * 4) 
    
    for j in range(len(err_single)):
        err_all += err_single[j]
    
    err_average = err_all / len(err_single)
    return err_single,err_average
 
if __name__=='__main__':
    #眼在手上
    # 棋盘格在相机下的位姿
    filename_RT_chess_to_cam = './findcheseboard_20230928-094722/RT_chess_to_cam.txt' # findcheseboard_20230928-110009/RT_chess_to_cam.txt
    # 相机在tcp下的位姿
    filename_RT_cam_to_end = './findcheseboard_20230928-094722/RT_cam_to_end.txt' # findcheseboard_20230928-110009/RT_cam_to_end.txt
    # tcp在机器人基坐标系下的位姿
    filename_RT_end_to_base = './findcheseboard_20230928-094722/RT_end_to_base.txt' # findcheseboard_20230928-110009/RT_end_to_base.txt
    # 选取的棋盘格上的点在机器人基坐标下的值(真实值)
    filename_real_base = 'real_chess_to_base_eye_in_hand.txt'
    # 选取的棋盘格上的点在相机坐标下的值(用于计算误差)
    filename_real_cam = 'real_chess_to_cam_eye_in_hand.txt'
    err_single,err_average = error_compute(filename_RT_chess_to_cam,filename_RT_cam_to_end,filename_RT_end_to_base,filename_real_base,filename_real_cam)
    print(f'眼在手上每个点的误差:\n{err_single}')
    print(f'眼在手上各个点的平均误差:\n{err_average}')

    #眼在手外
    filename_RT_chess_to_cam = './findcheseboard_20230928-102328/RT_chess_to_cam.txt'# 棋盘格在相机下的位姿
    filename_RT_cam_to_end = './findcheseboard_20230928-102328/RT_cam_to_end.txt' # 相机在机器人基坐标下的位姿
    filename_RT_end_to_base = './findcheseboard_20230928-102328/RT_end_to_base.txt'# 机器人原点在tcp坐标系下的位姿
    filename_real_base = 'real_chess_to_tcp_eye_to_hand.txt'#选取的棋盘格上的点在tcp坐标下的值(真实值)
    filename_real_cam = 'real_chess_to_cam_eye_to_hand.txt'#选取的棋盘格上的点在相机坐标下的值(用于计算)
    err_single,err_average = error_compute(filename_RT_chess_to_cam,filename_RT_cam_to_end,filename_RT_end_to_base,filename_real_base,filename_real_cam)
    print(f'眼在手外每个点的误差:\n{err_single}')
    print(f'眼在手外各个点的平均误差:\n{err_average}')