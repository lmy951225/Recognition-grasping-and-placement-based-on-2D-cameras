import numpy as np
import random
import argparse
import open3d as o3d
import copy
import math
import os


def rotation_matrix_to_quaternion(R):
    # 获取旋转矩阵的元素
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    # 计算四元素的元素
    q0 = np.sqrt(1 + r11 + r22 + r33) / 2
    q1 = (r32 - r23) / (4 * q0)
    q2 = (r13 - r31) / (4 * q0)
    q3 = (r21 - r12) / (4 * q0)

    return q0, q1, q2, q3

def rotation_matrix(x_angle, y_angle, z_angle):
    # 将角度转换为弧度
    x_angle = np.radians(x_angle)
    y_angle = np.radians(y_angle)
    z_angle = np.radians(z_angle)

    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x_angle), -np.sin(x_angle)],
                   [0, np.sin(x_angle), np.cos(x_angle)]])

    Ry = np.array([[np.cos(y_angle), 0, np.sin(y_angle)],
                   [0, 1, 0],
                   [-np.sin(y_angle), 0, np.cos(y_angle)]])

    Rz = np.array([[np.cos(z_angle), -np.sin(z_angle), 0],
                   [np.sin(z_angle), np.cos(z_angle), 0],
                   [0, 0, 1]])

    # 三个方向的旋转矩阵相乘得到最终的旋转矩阵
    R = np.dot(Rz, np.dot(Ry, Rx))

    return R

def quaternion_to_xyz(quaternion):
    q0, q1, q2, q3 = quaternion

    # Calculate rotation matrix elements
    r11 = 2 * q0 * q0 + 2 * q1 * q1 - 1
    r12 = 2 * q1 * q2 - 2 * q0 * q3
    r13 = 2 * q1 * q3 + 2 * q0 * q2
    r21 = 2 * q1 * q2 + 2 * q0 * q3
    r22 = 2 * q0 * q0 + 2 * q2 * q2 - 1
    r23 = 2 * q2 * q3 - 2 * q0 * q1
    r31 = 2 * q1 * q3 - 2 * q0 * q2
    r32 = 2 * q2 * q3 + 2 * q0 * q1
    r33 = 2 * q0 * q0 + 2 * q3 * q3 - 1

    # Calculate XYZ coordinates
    rx = np.arctan2(r32, r33)
    ry = np.arctan2(-r31, np.sqrt(r32 * r32 + r33 * r33))
    rz = np.arctan2(r21, r11)

    return rx, ry, rz


def npy2pcd(npy, ind=-1):
    colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    color = colors[ind] if ind < 3 else [random.random() for _ in range(3)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(npy)
    if ind >= 0:
        pcd.paint_uniform_color(color)
    return pcd


def fpfh(pcd, normals):
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=64)
    )
    return pcd_fpfh


def execute_fast_global_registration(source, target, source_fpfh, target_fpfh):
    distance_threshold = 1  # 0.01
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        ),
    )
    transformation = result.transformation
    estimate = copy.deepcopy(source)
    estimate.transform(transformation)
    R, t = transformation[:3, :3], transformation[:3, 3]
    return R, t, estimate


def fgr(source, target, src_normals, tgt_normals):
    source_fpfh = fpfh(source, src_normals)
    target_fpfh = fpfh(target, tgt_normals)
    R, t, estimate = execute_fast_global_registration(
        source=source, target=target, source_fpfh=source_fpfh, target_fpfh=target_fpfh
    )

    return R, t, estimate


def icp(source, target):
    max_correspondence_distance = 10000  # 0.5 in RPM-Net
    init = np.eye(4, dtype=np.float32)
    estimation_method = (
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source=source,
        target=target,
        init=init,
        max_correspondence_distance=max_correspondence_distance,
        estimation_method=estimation_method,
    )

    transformation = reg_p2p.transformation
    estimate = copy.deepcopy(source)
    estimate.transform(transformation)
    R, t = transformation[:3, :3], transformation[:3, 3]
    return R, t, estimate


def generate_rotation_x_matrix(theta):
    mat = np.eye(3, dtype=np.float32)
    mat[1, 1] = math.cos(theta)
    mat[1, 2] = -math.sin(theta)
    mat[2, 1] = math.sin(theta)
    mat[2, 2] = math.cos(theta)
    return mat


def generate_rotation_x_matrix(theta):
    mat = np.eye(3, dtype=np.float32)
    mat[1, 1] = math.cos(theta)
    mat[1, 2] = -math.sin(theta)
    mat[2, 1] = math.sin(theta)
    mat[2, 2] = math.cos(theta)
    return mat


def generate_rotation_y_matrix(theta):
    mat = np.eye(3, dtype=np.float32)
    mat[0, 0] = math.cos(theta)
    mat[0, 2] = math.sin(theta)
    mat[2, 0] = -math.sin(theta)
    mat[2, 2] = math.cos(theta)
    return mat


def generate_rotation_z_matrix(theta):
    mat = np.eye(3, dtype=np.float32)
    mat[0, 0] = math.cos(theta)
    mat[0, 1] = -math.sin(theta)
    mat[1, 0] = math.sin(theta)
    mat[1, 1] = math.cos(theta)
    return mat


def generate_random_rotation_matrix(angle1=-45, angle2=45):
    thetax, thetay, thetaz = np.random.uniform(angle1, angle2, size=(3,))
    matx = generate_rotation_x_matrix(thetax / 180 * math.pi)
    maty = generate_rotation_y_matrix(thetay / 180 * math.pi)
    matz = generate_rotation_z_matrix(thetaz / 180 * math.pi)
    return np.dot(matz, np.dot(maty, matx))


def generate_random_tranlation_vector(range1=-1, range2=1):
    tranlation_vector = np.random.uniform(range1, range2, size=(3,)).astype(np.float32)
    return tranlation_vector


def transform(pc, R, t=None):
    pc = np.dot(pc, R.T)
    if t is not None:
        pc = pc + t
    return pc


def get_bounding_box(point_cloud):
    x_min = np.min(point_cloud[:, 0])
    y_min = np.min(point_cloud[:, 1])
    z_min = np.min(point_cloud[:, 2])

    x_max = np.max(point_cloud[:, 0])
    y_max = np.max(point_cloud[:, 1])
    z_max = np.max(point_cloud[:, 2])

    return (x_min, y_min, z_min), (x_max, y_max, z_max)


def compute_scale(src_pcd, ref_pcd):
    # #尺度对齐
    (
        (src_x_min, src_y_min, src_z_min),
        (src_x_max, src_y_max, src_z_max),
    ) = get_bounding_box(np.asarray(src_pcd.points))
    (
        (dest_x_min, dest_y_min, dest_z_min),
        (dest_x_max, dest_y_max, dest_z_max),
    ) = get_bounding_box(np.asarray(ref_pcd.points))
    src_x_dist = src_x_max - src_x_min
    src_y_dist = src_y_max - src_y_min
    src_z_dist = src_z_max - src_z_min
    dest_x_dist = dest_x_max - dest_x_min
    dest_y_dist = dest_y_max - dest_y_min
    dest_z_dist = dest_z_max - dest_z_min
    
    scale_x=dest_x_dist / src_x_dist
    scale_y=dest_y_dist / src_y_dist
    scale_z=dest_z_dist / src_z_dist
    scale = np.mean([scale_x, scale_y, scale_z]) 

    return scale_x, scale_y, scale_z, scale

def xyz_scale(scale_xyz, pcd):
    # scale_xyz = [2.0, 1.5, 0.8]
    # 创建缩放变换矩阵
    scale_matrix = np.eye(4)
    scale_matrix[0, 0] = scale_xyz[0]
    scale_matrix[1, 1] = scale_xyz[1]
    scale_matrix[2, 2] = scale_xyz[2]

    # 对点云进行缩放
    scaled_point_cloud = pcd.transform(scale_matrix)
    return scaled_point_cloud

def pcd_align(
    src_pcd_path,
    ref_pcd_path,
    output_path,
    iters,
    scale_loss,
    save_txt,
    is_initRT,
    init_R,
):
    src_pcd = o3d.io.read_point_cloud(src_pcd_path)
    ref_pcd = o3d.io.read_point_cloud(ref_pcd_path)
    R_align = np.zeros([3, 3])
    T_align = np.array(ref_pcd.get_center()-src_pcd.get_center()) 
    scale_align=1
    scale_align_x = 1
    scale_align_y = 1
    scale_align_z = 1
    scale_x, scale_y, scale_z, scale = compute_scale(src_pcd, ref_pcd)

    for i in range(iters):
        
        pcd_scale = src_pcd.scale(scale, center=src_pcd.get_center())
        # pcd_scale = xyz_scale([scale_x, scale_y, scale_z], src_pcd)
        # print(scale, '+++')
        # #中心点对齐
        src_center = src_pcd.get_center()
        ref_center = ref_pcd.get_center()
        tx = np.array(ref_center - src_center)

        pcd_tx = copy.deepcopy(pcd_scale).translate(tx)

        # 粗配准
        if i == 0:  # 粗配准只进行一次
            if is_initRT:

                R_cu = np.array(init_R)
                R_cu = R_cu.reshape(3, 3)
                # x_angle = init_R[0]  # X轴旋转角度
                # y_angle = init_R[1]  # Y轴旋转角度
                # z_angle = init_R[2]  # Z轴旋转角度
                scale_align = scale
                scale_align_x = scale_x
                scale_align_y = scale_y
                scale_align_z = scale_z
                # R_cu = rotation_matrix(x_angle, y_angle, z_angle)
                pcd_gr=pcd_tx.rotate(R_cu)
                R_align=R_cu
                pcd_tx = pcd_gr

            else:
                # fgr粗配准
                # 计算点云的法向量
                pcd_tx.estimate_normals()
                ref_pcd.estimate_normals()

                # 获取点云的点坐标和法向量
                tx_points = np.asarray(pcd_tx.points)
                tx_normals = np.asarray(pcd_tx.normals)
                ref_points = np.asarray(ref_pcd.points)
                ref_normals = np.asarray(ref_pcd.normals)
                R_frg, t_frg, pred_ref_cloud = fgr(
                    source=npy2pcd(tx_points),
                    target=npy2pcd(ref_points),
                    src_normals=tx_normals,
                    tgt_normals=ref_normals,
                )
                pcd_tx = pred_ref_cloud
                pcd_tx.estimate_normals()
                ref_pcd.estimate_normals()

                # 获取点云的点坐标和法向量
                tx_points = np.asarray(pcd_tx.points)
                tx_normals = np.asarray(pcd_tx.normals)
                ref_points = np.asarray(ref_pcd.points)
                ref_normals = np.asarray(ref_pcd.normals)
                R_frg, t_frg, pred_ref_cloud = fgr(
                    source=npy2pcd(tx_points),
                    target=npy2pcd(ref_points),
                    src_normals=tx_normals,
                    tgt_normals=ref_normals,
                )
                R_align=R_frg
                pcd_tx = pred_ref_cloud                

        # ICP配准
        pcd_tx_center = -pcd_tx.get_center()
        ref_center = -ref_pcd.get_center()   
        src_icp = copy.deepcopy(pcd_tx).translate(pcd_tx_center)
        ref_icp = copy.deepcopy(ref_pcd).translate(ref_center)          
        R_icp, t_icp, estimate = icp(src_icp, ref_icp)

        
        pcd_rotate= pcd_tx.rotate(R_icp)
        pcd_rotate_center = pcd_rotate.get_center()
        ref_pcd_center =ref_pcd.get_center()

        tx1 = np.array(ref_pcd_center-ref_pcd_center)  

        pcd_rotate = pcd_rotate.translate(tx1)
        # estimate=pcd_rotate

        R_align=np.dot(R_icp, R_align)

        src_pcd = estimate
        scale_x_new, scale_y_new, scale_z_new, _ = compute_scale(src_pcd, ref_pcd)
        # (
        #     (src_x_min, src_y_min, src_z_min),
        #     (src_x_max, src_y_max, src_z_max),
        # ) = get_bounding_box(np.asarray(src_pcd.points))
      
        # src_x_dist = src_x_max - src_x_min
        # src_y_dist = src_y_max - src_y_min
        # src_z_dist = src_z_max - src_z_min
        
        # scale_x_new = dest_x_dist / src_x_dist
        # scale_y_new = dest_y_dist / src_y_dist
        # scale_z_new = dest_z_dist / src_z_dist

        
        if abs(scale_x - scale_x_new) < scale_loss and abs(scale_y - scale_y_new) < scale_loss and abs(scale_z - scale_z_new) < scale_loss:
            break

        scale_x = scale_x_new
        scale_y = scale_y_new
        scale_z = scale_z_new
        scale = np.mean([scale_x, scale_y, scale_z])
        scale_align *= scale
        scale_align_x *= scale_x
        scale_align_y *= scale_y
        scale_align_z *= scale_z


        
        # print('scale:::', scale_align)

        # o3d.io.write_point_cloud(
        #         os.path.join(output_path, "estimate.ply"),
        #         # pcd_gr,
        #         estimate,
        #         write_ascii=False,
        #         compressed=False,
        #         print_progress=False,
        #     )

    if save_txt:
        src_pcd = o3d.io.read_point_cloud(src_pcd_path)
        ref_pcd = o3d.io.read_point_cloud(ref_pcd_path)
        src_pcd1 = copy.deepcopy(src_pcd)
        # scale_out_x, scale_out_y, scale_out_z, scale_out = compute_scale(src_pcd, ref_pcd)
        # print('number of iters::', i, '   x, y, z:::', scale_out_x, scale_out_y, scale_out_z, scale_out)
        pcd_scale = src_pcd.scale(scale_align, center=src_pcd.get_center())
        # pcd_scale = xyz_scale([scale_align_x, scale_align_y, scale_align_z], src_pcd)
        pcd_scale = pcd_scale.rotate(R_align)

        src_center = pcd_scale.get_center()
        ref_center = ref_pcd.get_center()
        # T_align = np.array(ref_center - src_center) 

        src_xyz_load = np.asarray(pcd_scale.points)        
        (src_x_min, src_y_min, src_z_min), (src_x_max, src_y_max, src_z_max)=get_bounding_box(src_xyz_load)  
        # src_x_dist=src_x_max-src_x_min
        # src_y_dist=src_y_max-src_y_min
        # src_z_dist=src_z_max-src_z_min
        src_x_center=(src_x_max+src_x_min)/2
        src_y_center=(src_y_max+src_y_min)/2
        src_z_center=(src_z_max+src_z_min)/2        
        src_center=np.array([src_x_center,src_y_center,src_z_center])

        dest_xyz_load = np.asarray(ref_pcd.points)        
        (dest_x_min, dest_y_min, dest_z_min), (dest_x_max, dest_y_max, dest_z_max)=get_bounding_box(dest_xyz_load)  
        # dest_x_dist=dest_x_max-dest_x_min
        # dest_y_dist=dest_y_max-dest_y_min
        # dest_z_dist=dest_z_max-dest_z_min
        dest_x_center=(dest_x_max+dest_x_min)/2
        dest_y_center=(dest_y_max+dest_y_min)/2
        dest_z_center=(dest_z_max+dest_z_min)/2
        dest_center = np.array([dest_x_center,dest_y_center,dest_z_center]) 
        # src_x_dist = src_x_max - src_x_min
        # src_y_dist = src_y_max - src_y_min
        # src_z_dist = src_z_max - src_z_min
        # dest_x_dist = dest_x_max - dest_x_min
        # dest_y_dist = dest_y_max - dest_y_min
        # dest_z_dist = dest_z_max - dest_z_min
        
        # scale_x = dest_x_dist / src_x_dist
        # scale_y = dest_y_dist / src_y_dist
        # scale_z = dest_z_dist / src_z_dist

        # scale = np.mean([scale_x, scale_y, scale_z])#*0.97
        print('x1, y1, z1:::', scale_align_x, scale_align_y, scale_align_z)

        # scale_align *= scale

        # pcd_scale = pcd_scale.scale(scale_align, center=pcd_scale.get_center())

        
        T_align = np.array(dest_center - src_center) 
        pcd_gr = copy.deepcopy(pcd_scale).translate(T_align)
        gr_center = pcd_gr.get_center()
        scale_out_x, scale_out_y, scale_out_z, scale_out = compute_scale(src_pcd1, pcd_gr)
        print('number of iters::', i, '   x, y, z:::', scale_out_x, scale_out_y, scale_out_z, scale_out)
                
        # scale_t = np.repeat(scale_align, 3)
        scale_t = np.array([scale_out_x, scale_out_y, scale_out_z]) 
        scale_t = np.expand_dims(scale_t, axis=1)
        sup_t = np.expand_dims(T_align, axis=1)
        gr_rt = np.hstack((R_align, sup_t,scale_t))
        np.savetxt(os.path.join(output_path, "align_1019.txt"), gr_rt, fmt="%f", delimiter=" ")
        
    r_xyz = (
        np.array(quaternion_to_xyz(rotation_matrix_to_quaternion(R_align)))
        / math.pi
        * 180
    )
    
    o3d.io.write_point_cloud(
        os.path.join(output_path, "align_1019.ply"),
        pcd_gr,
        # estimate,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )
    print("SCALE-XYZ", scale_align)
    print("R-XYZ", r_xyz[0], r_xyz[1], r_xyz[2])
    print("T-XYZ", T_align[0], T_align[1], T_align[2])
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D align")
    parser.add_argument(
        "--src-pcd",
        type=str,
        default="/home/jingjing/Documents/align/PLY.ply",
        help="path for source pcd files",
    )
    parser.add_argument(
        "--ref-pcd",
        type=str,
        default="/home/jingjing/Documents/crop/byd1018.ply",
        help="path for reference pcd files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/jingjing/Documents/align/",
        help="save path for align pcd files",
    )
    parser.add_argument("--iters", type=int, default=100, help="iterations")
    parser.add_argument("--scale-loss", type=int, default=0.0001, help="scale loss")
    parser.add_argument("--save-txt", type=bool, default=True, help="save RT to txt")
    parser.add_argument("--is-initRT", type=bool, default=True, help="init RT flag")
    parser.add_argument(
        "--init-R",
        type=list,  #[-0.979, 0.191, 0.075, 0.190, 0.710, 0.679, 0.076, 0.678, -0.731], 
        default=[0.016, 0.973, 0.230, 0.949, -0.087, 0.303, 0.315, 0.213, -0.925],#[-0.943, 0.242, -0.227, 0.068, 0.810, 0.583, 0.325, 0.535, -0.780],#[-0.859,-0.494,-0.135,-0.482,0.869,-0.112,0.172,-0.031,-0.985], #[-0.979, 0.203, 0.021, 0.174, 0.774, 0.609, 0.107, 0.600, -0.793],
        help="init R",
    )

    args = parser.parse_args()

    pcd_align(
        args.src_pcd,
        args.ref_pcd,
        args.output_dir,
        args.iters,
        args.scale_loss,
        args.save_txt,
        args.is_initRT,
        args.init_R,
    )
