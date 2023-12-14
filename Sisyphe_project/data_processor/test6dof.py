import numpy as np
import os
import csv
from utils_rot import *
np.set_printoptions(suppress=True)
from scipy.spatial.transform import Rotation

def convert_to_4x4(translation, euler_angles):
    rotation = Rotation.from_euler('xyz', euler_angles).as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    return transformation_matrix

# T_jaw_2_tcp=np.array([
#     [1,0,0,-0.77648664],
#     [0,-1,0,1.41247193],
#     [0,0,-1,238.30122821],
#     [0,0,0,1]
# ])

v_jaw_2_tcp = np.array([-0.77648664, 1.41247193, 238.30122821, 3.14159265, -0., 0.])
# v_jaw_2_tcp = np.array([0, 0, 238.30122821, 3.14159265, -0., 0.])
# 90
# v_cam_2_tcp = np.array([56.68305607, 0.50922745, 190.8758467,
#                         -0.00607595 - 0.2/180*math.pi, -0.01239557 - 0.2/180*math.pi , -1.59065785])

# v_cam_2_tcp = np.array([55.8011, 2.1644, 188.9487,
#                         -0.016672315774268916 + 0.4/180*math.pi, -0.00792426291428139 - 2.5/180*math.pi, -1.5923833610502092])

# v_cam_2_tcp = np.array([52.8011, 2.1644, 188.9487,
#                         -0.016672315774268916 , -0.00792426291428139 , -1.5923833610502092])

v_cam_2_tcp = np.array([56.68305607, 0.50922745, 190.8758467,
                        -0.00607595, -0.01239557, -1.59065785])

# v_cam_2_tcp = np.array([ -0.02158468 ,  0.99962521  , 0.01683869 , 52.8011    ],
#                        [ -0.99973562 , -0.02171443  , 0.00756137 ,  2.1644    ],
#                        [  0.00792418 , -0.01667102  , 0.99982963 ,188.9487    ],
#                        [  0.         ,  0.         ,  0.         ,  1.        ])

T_jaw_2_tcp = convert_to_4x4(v_jaw_2_tcp[:3], v_jaw_2_tcp[3:])
T_cam_2_tcp = convert_to_4x4(v_cam_2_tcp[:3], v_cam_2_tcp[3:])

# print(T_jaw_2_tcp, T_cam_2_tcp)

# T_cam_2_tcp=np.array([[ -0.01985869  , 0.99978282  , 0.00632089 , 56.68305607],
#  [ -0.99972596 , -0.01993514 ,  0.01227191 ,  0.50922745],
#  [  0.01239525 , -0.00607545  , 0.99990472 ,190.8758467 ],
#  [  0.     ,      0.   ,        0.    ,       1.        ]])

##

def read_tcp_trafectory(path_circle, path_sphere):

    traj_list=[]
    # opening the CSV file
    with open(path_circle, mode='r') as file:
        # reading the CSV file
        csvFile = csv.reader(file)
        # displaying the contents of the CSV file
        # for lines in csvFile:
        for idx,lines in enumerate(csvFile):
            if idx%2 == 0:
                continue
            else:
                # if len(lines)>1:
                temp=[float(item) for item in lines[:6]]
                traj_list.append(temp)
                # print(lines)
    with open(path_sphere, mode='r') as file:
        # reading the CSV file
        csvFile = csv.reader(file)
        # displaying the contents of the CSV file
        # for lines in csvFile:
        for idx,lines in enumerate(csvFile):
            if idx%2 == 0:
                continue
            else:
                # if len(lines)>1:
                temp=[float(item) for item in lines[:6]]
                traj_list.append(temp)
                # print(lines)
    return traj_list


if __name__ == '__main__':
    path_circle = '/media/jingjing/56a1e372-b7ba-43a2-8adc-ce445bc31fff/jingjing/share/data/bydGT_1107/all/trajectory_circle_315.csv'
    path_sphere = '/media/jingjing/56a1e372-b7ba-43a2-8adc-ce445bc31fff/jingjing/share/data/bydGT_1107/all/trajectory_sphere_315.csv'
    images_path = '/media/jingjing/56a1e372-b7ba-43a2-8adc-ce445bc31fff/jingjing/share/data/bydGT_1107/dataset_315/images'
    save_path = '/media/jingjing/56a1e372-b7ba-43a2-8adc-ce445bc31fff/jingjing/share/data/bydGT_1107/dataset_315/images_trans.txt'
    
    T_grasp_in_obj=np.loadtxt('gt/data2/T_grasp_in_obj.txt')
    tcp_grasp=[410.549,341.133,268.518,162,0,315]#[435.336,281.291,268.518,162,0,270] #[410.549,221.449,268.518,162,0,225]#[350.707,196.662,268.518,162,0,180]  # [266.078,281.291,268.518,162,0,90]  #[343.569,378.362,268.518,162,0,0]#real
    T_tcp_grasp_in_w=get_T_from_6dof(tcp_grasp)
    T_jaw_grasp_in_w=T_tcp_grasp_in_w@T_jaw_2_tcp
    ### T_jaw_grasp_in_w is T_grasp_in_w
    T_grasp_in_w=T_jaw_grasp_in_w
    ##step_1: get obj in world
    T_obj_in_w=T_grasp_in_w@np.linalg.inv(T_grasp_in_obj)
    ########

    traj_list = read_tcp_trafectory(path_circle, path_sphere)

    sfiles = os.listdir(images_path)
    sind = [int(file.split('_')[1]) for file in sfiles]
    cam_pose_list = []
    for idx_tcp,tcp in enumerate(traj_list):
        ##step_2: get cam in world
        T_tcp_in_w=get_T_from_6dof(tcp)
        T_cam_in_w=T_tcp_in_w@T_cam_2_tcp
        print(T_cam_in_w[:, 3])
        #########
        ##step_3: get obj in cam
        T_obj_in_cam=np.linalg.inv(T_cam_in_w)@T_obj_in_w

        nm = sfiles[sind.index(idx_tcp+1)]
        np.savetxt('gt/gt_pose/'+nm+'.txt',T_obj_in_cam)
        # print(idx_tcp)
        cam_pose = transform_matrix_to_pose(T_obj_in_cam).tolist()
        cam_pose.append(nm)
        # print(cam_pose)
        cam_pose_1 = ' '.join(list(map(str, cam_pose)))
        cam_pose_list.append(cam_pose_1)


    with open(save_path, 'w') as f:
        for line in cam_pose_list:
            f.writelines(str(line))
            f.write('\n')




####### post process:
# T_est_grasp_in_w=T_est_obj_in_w@T_grasp_in_obj