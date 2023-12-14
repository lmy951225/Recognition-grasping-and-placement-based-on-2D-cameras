import numpy as np
import cv2
np.set_printoptions(suppress=True)

#中心点 x:80  y:75
# center_point = np.array([666.786, 351.550, 296.557, 180, 0, 0 ])

def get_center_point(filename,z=296.557):
    '''
    param: filename:托盘图片路径
    param: z:放置工件时的tcp高度
    return: center_point:托盘中心点对应的tcp位姿
    '''
    # 读取图片估计得到中心点位置
    img = cv2.imread(filename)
    center_point = np.array([666.786, 351.550, z, 180, 0, 0 ])
    return center_point


def get_Place_tcp(center_point,x_offset,y_offset,nx,ny):
    '''
    param: center_point:托盘中心点对应的tcp位姿
    param: x_offset:工件间的x间距
    param: y_offset:工件间的y间距
    param: nx:行数
    param: ny:列数
    return: tcp_pose:放置的tcp位姿
    '''

    # r=(n+1)//2
    # print(-n//2)
    unit_x = np.arange(-(nx//2),(nx+1)//2)
    unit_y = np.arange(-(ny//2),(ny+1)//2)
    xv,yv = np.meshgrid(unit_x*x_offset,unit_y*y_offset)
    xv = xv.flatten()
    yv = yv.flatten()
    tcp_pose = []
    #1号位-9号位
    for x,y in zip(xv,yv):
        # print(x,y)
        tcp_pose.append(np.array([center_point[0] + x, center_point[1] + y, center_point[2], center_point[3], center_point[4], center_point[5]]))
    # print(tcp_pose)
    return tcp_pose

def tcp_chose(tcp_pose,nx,ny,flag='x'):
    '''
    param:tcp_pose:下料的tcp位姿
    param: nx:行数
    param: ny:列数
    flag:'x','y','xs','ys'分别代表沿x轴,y轴,x轴S型,y轴S型放置
    '''
    if flag == 'x':
        tcp_pose = tcp_pose
    elif flag == 'y':
        tcp_pose = np.array(tcp_pose).reshape(nx,ny,6).transpose(1,0,2).reshape(-1,6).tolist()
    elif flag == 'xs':
        tcp_pose = np.array(tcp_pose).reshape(nx,ny,6)
        tcp_pose_new=np.zeros_like(tcp_pose)
        flag_reverse=1
        for id_x in range(nx):
            if flag_reverse == -1:
                tcp_pose_new[id_x,:,:]=tcp_pose[id_x,::-1,:]
            else:
                tcp_pose_new[id_x,:,:]=tcp_pose[id_x,:,:]
            flag_reverse *= -1
        tcp_pose = tcp_pose_new.reshape(-1,6).tolist()
    elif flag == 'ys':
        tcp_pose = np.array(tcp_pose).reshape(nx,ny,6).transpose(1,0,2)
        tcp_pose_new=np.zeros_like(tcp_pose)
        flag_reverse=1
        for id_x in range(nx):
            if flag_reverse == -1:
                tcp_pose_new[id_x,:,:]=tcp_pose[id_x,::-1,:]
            else:
                tcp_pose_new[id_x,:,:]=tcp_pose[id_x,:,:]
            flag_reverse *= -1
        tcp_pose = tcp_pose_new.reshape(-1,6).tolist()
    return tcp_pose      


if __name__ == "__main__":
    center_point = np.array([666.786, 351.550, 296.557, 180, 0, 0 ])
    # print(*center_point)
    nx = 4
    ny = 4
    x_offset = 80
    y_offset = 75
    tcp = get_Place_tcp(center_point,x_offset,y_offset,nx,ny)
    # for i in tcp:
    #     print(i)
    tcp_pose = tcp_chose(tcp,nx,ny,'xs')
    for i in tcp_pose:
        print(*i)