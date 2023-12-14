"""
@Author      :   XiaoZhiheng
@Time        :   2023/01/31 10:34:06
"""

from copy import deepcopy
import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from typing import Dict, List, Tuple
from math import sin, cos, pi
import numpy as np 
import yaml
import json
import xml.etree.ElementTree as ET
import open3d as o3d
import os 
# from containers.RobotPose import RobotPose
from scipy.spatial.transform import Rotation as R 

def rotation_with_fixed_axis(theta, rotation_axis):

    t_x, t_y, t_z = rotation_axis[0]
    Trans = np.array([
        [1, 0, 0, t_x],
        [0, 1, 0, t_y],
        [0, 0, 1, t_z],
        [0, 0, 0, 1]
    ])
    Rot = np.array([
        [cos(theta), -sin(theta), 0, 0],
        [sin(theta),  cos(theta), 0, 0],
        [         0,           0, 1, 0],
        [         0,           0, 0, 1]
    ])

    Inv_Trans = np.array([
        [1, 0, 0, -t_x],
        [0, 1, 0, -t_y],
        [0, 0, 1, -t_z],
        [0, 0, 0, 1]
    ])

    return Trans @ Rot @ Inv_Trans

def homogeneous_matrix_to_euler(matrix, to_degree:bool=True, to_millimeters:bool=True):
    
    rot = R.from_matrix(matrix[:3,0:3])
    Rx_, Ry_, Rz_ = rot.as_euler("xyz", to_degree)
    X_, Y_, Z_ = matrix[0,3], matrix[1,3], matrix[2,3]
    
    if to_millimeters:
        X_, Y_, Z_ = X_ * 1000, Y_ * 1000, Z_ * 1000 

    return np.array([X_, Y_, Z_, Rx_, Ry_, Rz_])


def rpy_to_rotation_matrix(Rx_, Ry_, Rz_, is_degree:bool=False):

    if is_degree:
        Rx_, Ry_, Rz_  = np.array([Rx_, Ry_, Rz_]) / 180 * pi
    T_z = np.array(
        [[cos(Rz_), -sin(Rz_), 0, 0],
            [sin(Rz_),  cos(Rz_), 0, 0],
            [       0,         0, 1, 0],
            [       0,         0, 0, 1]]
    )
    T_y = np.array(
        [[ cos(Ry_), 0, sin(Ry_), 0],
            [        0, 1,        0, 0],
            [-sin(Ry_), 0, cos(Ry_), 0],
            [       0,  0,        0, 1]]
    )
    T_x = np.array(
        [[1,        0,         0, 0],
            [0, cos(Rx_), -sin(Rx_), 0],
            [0, sin(Rx_),  cos(Rx_), 0],
            [0,        0,         0, 1]]
    )
    rotation_matrix = T_z @ T_y @ T_x 
    
    return rotation_matrix

def euler_to_homogeneous_matrix(xyzrxryrz:np.ndarray, is_degree:bool=True, is_millimeters=True):

    X_, Y_, Z_, Rx_, Ry_, Rz_ = xyzrxryrz
    rotation_matrix = rpy_to_rotation_matrix(Rx_, Ry_, Rz_, is_degree)
    if is_millimeters:
        X_, Y_, Z_ = X_ / 1000, Y_ / 1000, Z_ / 1000
    T_shift = np.array(
        [[1, 0, 0, X_],
            [0, 1, 0, Y_],
            [0, 0, 1, Z_],
            [0, 0, 0,  1]]
    )
    tran_matrix = T_shift @ rotation_matrix
    
    return tran_matrix


def read_yaml(yaml_path: str) -> Dict:
    """读yaml文件
    Args:
        yaml_path: yaml文件的地址
    Returns:
        file: 读取的yaml文件信息
    """

    with open(yaml_path, 'r', encoding='utf-8') as f:
        file = yaml.load(f, Loader=yaml.FullLoader)

    return file

def read_json(json_path: str) -> Dict:

    with open(json_path, "r") as f:
        file = json.load(f)

    return file 

def virtual2real(radians: np.ndarray, manipulator_type: str, isToAngle:bool=True)-> np.ndarray: 

    if isToAngle:
        angles = radians / pi * 180
    else:
        angles = radians
    if manipulator_type.startswith("LR_Mate_200iD"): 
        angles[2] = -angles[2] - angles[1]
        angles[3] = -angles[3]
        angles[4] = -angles[4]
        angles[5] = -angles[5] 
    elif manipulator_type == "P7A_900":
        angles[1] = -angles[1]
        angles[2] = -angles[2]
        angles[4] = -angles[4]
    else:
        raise ValueError("Unknow manipulator type: {}.".format(manipulator_type))

    return angles

def real2virtual(angles: np.ndarray, manipulator_type: str, isToRadians:bool=True)-> np.ndarray:

    if isToRadians:
        radians = angles * pi / 180
    else:
        radians = angles
    
    if manipulator_type.startswith("LR_Mate_200iD"):
        
        radians[2] = -radians[2] - radians[1]
        radians[3] = -radians[3]
        radians[4] = -radians[4]
        radians[5] = -radians[5]
        
    elif manipulator_type == "P7A_900":
        radians[1] = -radians[1]
        radians[2] = -radians[2]
        radians[4] = -radians[4]
    else:
        raise ValueError("Unknow manipulator type: {}.".format(manipulator_type))

    return radians

def generate_stard_obstacles_config(obstacles_config:Dict) -> Dict:

    obstacles:Dict = obstacles_config["obstacles"]
    for obs in obstacles.values():

        pose_b = obs["pose_b"]
        new_pose_b = [[1, 0, 0, pose_b[0]],
                      [0, 1, 0, pose_b[1]],
                      [0, 0, 1, pose_b[2]],
                      [0, 0, 0,         1]]
        obs["pose_b"] = new_pose_b
    
    return obstacles_config

def transform_xyzrxryrz(xyzrxryrz: np.ndarray) -> np.ndarray: 

    xyzrxryrz[:3] = xyzrxryrz[:3] / 1000
    xyzrxryrz[3:] = xyzrxryrz[3:] / 180 * pi

    return xyzrxryrz


def analysis_urdf_file(urdf_path: str) -> Tuple[List[str], List[str], List[np.ndarray]]:

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links = root.findall('link')
    stl_path_list = []
    link_name_list = []
    mesh_coordinate_shift_list = []
    for link in links:
        try:
            x, y, z = link.find('visual').find('origin').attrib['xyz'].split()
            xyz = np.array([float(x), float(y), float(z)])
            R, P, Y = link.find('visual').find('origin').attrib['rpy'].split()
            rotation_M = rpy_to_rotation_matrix(float(R), float(P), float(Y))
            xyzw = rotation_M.T @ np.insert(xyz, 3, 1)
            mesh_coordinate_shift_list.append(xyzw[0:3])
            stl_path = link.find('visual').find('geometry').find('mesh').attrib['filename']
            stl_path_list.append(stl_path)
            link_name_list.append(link.attrib["name"])
        except:
            pass 
    
    return root.attrib["name"], link_name_list, stl_path_list, mesh_coordinate_shift_list

def get_mesh_info(stl_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if os.path.exists(stl_path):
        mesh_or = o3d.io.read_triangle_mesh(stl_path)
    elif os.path.exists(stl_path.replace(".stl", ".STL")):
        mesh_or = o3d.io.read_triangle_mesh(stl_path.replace(".stl", ".STL"))
    else:
        raise ValueError("Can't find {}".format(stl_path))
    mesh = mesh_or.simplify_vertex_clustering(
        voxel_size = max(mesh_or.get_max_bound() - mesh_or.get_min_bound()) / 40,
        contraction = o3d.geometry.SimplificationContraction.Average
    )
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    return vertices, triangles

def generate_link_config(robot_urdf_path: str) -> Dict:

    def modify_coordinate_in_self_frame(
        robot_name: str, link_name: str, vertices:np.ndarray
        ) -> np.ndarray: 
        """The meaning of self frame is a DH coordinate system defined by ourself.

        Args:
            link_name (str): 
            vertices (np.ndarray): 

        Returns:
            np.ndarray : 
        """
        
        new_vertices = deepcopy(vertices)
        if robot_name.startswith("FANUC_LR_Mate_200iD"): 
            if link_name == "Link2":
                new_vertices[:, 0] =  vertices[:, 0]
                new_vertices[:, 1] = -vertices[:, 2]
                new_vertices[:, 2] =  vertices[:, 1]
            elif link_name == "Link3" or link_name == "Link5":
                new_vertices[:, 0] =  vertices[:, 2]
                new_vertices[:, 1] =  vertices[:, 0]
                new_vertices[:, 2] =  vertices[:, 1]
            elif link_name == "Link4":
                new_vertices[:, 0] =  vertices[:, 2]
                new_vertices[:, 1] = -vertices[:, 1]
                new_vertices[:, 2] =  vertices[:, 0]
            elif link_name == "Link6":
                new_vertices[:, 0] =  vertices[:, 2]
                new_vertices[:, 1] = -vertices[:, 1]
                new_vertices[:, 2] =  vertices[:, 0]
                
        elif robot_name == "P7A_900":
            if link_name == "Link1":
                new_vertices[:, 0] =  vertices[:, 0]
                new_vertices[:, 1] =  vertices[:, 2]
                new_vertices[:, 2] =  vertices[:, 1]
            elif link_name == "Link2":
                new_vertices[:, 0] =  vertices[:, 0]
                new_vertices[:, 1] = -vertices[:, 1]
                new_vertices[:, 2] = -vertices[:, 2]      
            elif link_name == "Link3":
                new_vertices[:, 0] =  vertices[:, 1]
                new_vertices[:, 1] =  vertices[:, 0]
                new_vertices[:, 2] = -vertices[:, 2]
            elif link_name == "Link4":
                new_vertices[:, 0] =  vertices[:, 1]
                new_vertices[:, 1] =  vertices[:, 2]
                new_vertices[:, 2] =  vertices[:, 0] + 0.1915
            elif link_name == "Link5":
                new_vertices[:, 0] = -vertices[:, 1]
                new_vertices[:, 1] =  vertices[:, 0]
                new_vertices[:, 2] =  vertices[:, 2] 
            elif link_name == "Link6":
                new_vertices[:, 0] =  vertices[:, 1]
                new_vertices[:, 1] =  vertices[:, 2]
                new_vertices[:, 2] =  vertices[:, 0] 
            
        return new_vertices
    
    urdf_path = os.path.join(robot_urdf_path, "robot_link.urdf")
    robot_name, link_name_list, stl_path_list, mesh_coordinate_shift_list = analysis_urdf_file(urdf_path)
    links_config = dict()
    
    for link_name, stl_path, mesh_coordinate_shift in zip(link_name_list, stl_path_list, mesh_coordinate_shift_list):
   
        stl_path = os.path.join(robot_urdf_path, stl_path)
        vertices, triangles = get_mesh_info(stl_path)
        vertices += mesh_coordinate_shift
        vertices = modify_coordinate_in_self_frame(robot_name, link_name, vertices)
        
        # mesh = trans_mesh(vertices=vertices, triangles=triangles, trans_matrix=np.identity(4))
        # FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([mesh, FOR1])
        
        links_config[link_name] = {
            "type":"Box",
            "color":"bwr",
            "vertices": vertices,
            "triangles": triangles
        }
    return {"Links": links_config}

def generate_end_effector_config(end_effector_stl_path: str) -> Dict:
    
    vertices, triangles = get_mesh_info(end_effector_stl_path)
    vertices /= 1000
    end_effector_config = {
        "type":"Box",
        "color":"bwr",
        "vertices": vertices,
        "triangles": triangles
    }
    
    return {"end_effector": end_effector_config}

def generate_obstacles_config(obstacles_stl_path_list: List[str], keyword: str) -> Dict:
    
    osbtacles_config = dict()
    for stl_path in obstacles_stl_path_list:
        
        name = os.path.basename(stl_path).split(".")[0]
        if name.startswith("Product"):
            key = "product"
        elif name.startswith("RFoundation"):
            key = "rfoundation"
        elif name.startswith("TFoundation"):
            key = "tfoundation"
        elif name.startswith("Base"):
            key = "base"
        elif name.startswith("Screen"):
            key = "screen"
        else:
            key = name
        vertices, triangles = get_mesh_info(stl_path)
        vertices /= 1000
        osbtacles_config[key] = {
            "type":"Box",
            "color":"Reds_r",
            "vertices": vertices,
            "triangles": triangles
        }
    
    if keyword == "static":
        return {"static_osbtacles": osbtacles_config}
    elif keyword == "dynamic":
        return {"dynamic_osbtacles": osbtacles_config}
    else:
        raise Exception("The given keyword does not exist.")

def sort_by_distance(sequence:np.ndarray, target:float) -> np.ndarray:

    distance = (sequence - target) ** 2
    min_index = np.argsort(distance)
    new_sequence = sequence[min_index]
    
    return new_sequence

def generate_range(bounder, num):
        
    min_ = bounder[0]
    max_ = bounder[1] 
    if num <= 1:
        return np.array([min_], np.float64)
    step = (max_ - min_) / (num-1)

    return np.array([min_+i*step  for i in range(num)], np.float64)

def rotate_xyzrxryrz(xyzrxryrz_or: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:

    xyzrxryrz = deepcopy(xyzrxryrz_or)
    # xyzrxryrz = transform_xyzrxryrz(xyzrxryrz)
    Tran_matrix = euler_to_homogeneous_matrix(xyzrxryrz)
    rotated_tran_matrix = rotation_matrix @ Tran_matrix
    rotated_xyzrxryrz = homogeneous_matrix_to_euler(rotated_tran_matrix)
    # rotated_xyzrxryrz[0:3] = rotated_xyzrxryrz[0:3] * 1000

    return rotated_xyzrxryrz

# def rotate_robot_poses(robot_poses: List[RobotPose], rotation_matrix: np.ndarray) -> List[RobotPose]:
        
    for robot_pose in robot_poses:
        xyzrxryrz = robot_pose.xyzrxryrz
        rotated_xyzrxryrz = rotate_xyzrxryrz(xyzrxryrz, rotation_matrix)
        robot_pose.xyzrxryrz = rotated_xyzrxryrz

    return robot_poses


def trans_mesh(vertices: np.ndarray, triangles:np.ndarray, trans_matrix:np.ndarray):
    """

    Args:
        vertices (np.ndarray): Vertices of triangular mesh.
        triangles (np.ndarray): Index of triangles.
        trans_matrix (np.ndarray): Matrix used to describe pose.

    Returns:
        mesh: Open3d geometry triangleMesh.
    """
    
    if trans_matrix is not None:
        w = np.ones((len(vertices),1))
        vertices_w = np.concatenate([vertices, w], axis=1)
        vertices_w = vertices_w @ trans_matrix.T
        vertices = vertices_w[:, 0:3]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    return mesh

def read_txt(file_path:str) -> List:
    
    with open(file_path) as f:
        data = []
        lines = f.readlines()
        for line in lines:
            split_line = line.split(" ")
            l_list = [eval(i) for i in split_line]
            data.append(l_list)
            
    return data

def read_csv(file_path:str) -> List:
    
    import pandas as pd
    data = pd.read_csv(file_path)
    row_list = data.values.tolist()
    
    return row_list

if __name__ == "__main__":

    path_ = "/home/xzh/host_dir/3DModel/RobotURDF/LR_Mate_200iD_7L/Meshes"
    name_list = ["BaseLink.stl", "Link1.stl", "Link2.stl", "Link3.stl", "Link4.stl", "Link5.stl", "Link6.stl"]
    mesh_list = []
    for name in name_list:
        mesh = o3d.io.read_triangle_mesh(os.path.join(path_, name))
        bbox = mesh.get_axis_aligned_bounding_box()
        print(bbox)
        mesh_list.append(mesh)
        FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        mesh_list.append(FOR1)
        o3d.visualization.draw_geometries([mesh, FOR1])
    o3d.visualization.draw_geometries(mesh_list)
    
    pass 





