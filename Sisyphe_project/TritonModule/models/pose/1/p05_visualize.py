import os
import logging

import cv2
import numpy as np
import open3d as o3d


def show_pose(pose_est, object_info, pose):
    rigid_body = np.array(
        [[object_info["max_x"], object_info["max_y"], object_info["max_z"]],
         [object_info["max_x"], object_info["min_y"], object_info["max_z"]],
         [object_info["min_x"], object_info["min_y"], object_info["max_z"]],
         [object_info["min_x"], object_info["max_y"], object_info["max_z"]],
         [object_info["max_x"], object_info["max_y"], object_info["min_z"]],
         [object_info["max_x"], object_info["min_y"], object_info["min_z"]],
         [object_info["min_x"], object_info["min_y"], object_info["min_z"]],
         [object_info["min_x"], object_info["max_y"], object_info["min_z"]]])
    rigid_body *= 1000
    rotation_matrix = pose_est[:, 0:3]

    # 进行旋转
    position = np.dot(rotation_matrix, rigid_body.T).T + pose_est[:, 3]
    position_w = np.dot(pose[:3, :3], position.T).T + pose[:3, 3]
    logging.info("pose::::", pose)
    center_R_w = np.dot(pose[:3, :3], rotation_matrix)
    center_t_w = np.dot(pose[:3, :3], pose_est[:, 3].T) + pose[:3, 3]

    return position, position_w, center_R_w, center_t_w

def show_result(img, pose, position, position_w, center_t_w, cam_k, name="pr"):
    result = "result"
    if not os.path.exists(result):
        os.makedirs(result)
    position_w2 = np.append(position_w, center_t_w.reshape(-1, 3), axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(position_w2)
    # 将点云保存为PLY文件
    o3d.io.write_point_cloud("./models/output.ply", pcd)

    u = cam_k[0, 0] * pose[0, 3] / pose[2, 3] + cam_k[0, 2]
    v = cam_k[1, 1] * pose[1, 3] / pose[2, 3] + cam_k[1, 2]

    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 2

    # 画点
    point = (int(u), int(v))  # 点的坐标。画点实际上就是画半径很小的实心圆。
    cv2.circle(img, point, point_size, point_color, thickness)

    points = []
    for i in range(len(position)):
        pose0 = position[i]
        # print(pose0)
        u = cam_k[0, 0] * pose0[0] / pose0[2] + cam_k[0, 2]
        v = cam_k[1, 1] * pose0[1] / pose0[2] + cam_k[1, 2]
        points.append([u, v])

    point_size = 1
    point_color = point_color
    thickness = 2
    for i in range(len(points)):
        point = (int(points[i][0]), int(points[i][1])
                 )  # 点的坐标。画点实际上就是画半径很小的实心圆。

        cv2.circle(img, point, point_size, point_color, thickness)
        if (i + 1) % 4 == 0 and i > 0:
            # pass
            cv2.line(img,
                     point, (int(points[i - 3][0]), int(points[i - 3][1])),
                     point_color,
                     thickness=1)
        else:
            point2 = (int(points[i + 1][0]), int(points[i + 1][1])
                      )  # 点的坐标。画点实际上就是画半径很小的实心圆。
            cv2.line(img, point, point2, point_color, thickness=1)
    cv2.line(img, (int(points[0][0]), int(points[0][1])),
             (int(points[4][0]), int(points[4][1])),
             point_color,
             thickness=1)
    cv2.line(img, (int(points[1][0]), int(points[1][1])),
             (int(points[5][0]), int(points[5][1])),
             point_color,
             thickness=1)
    cv2.line(img, (int(points[2][0]), int(points[2][1])),
             (int(points[6][0]), int(points[6][1])),
             point_color,
             thickness=1)
    cv2.line(img, (int(points[3][0]), int(points[3][1])),
             (int(points[7][0]), int(points[7][1])),
             point_color,
             thickness=1)

    if name == 'crawl':
        cv2.imwrite("/models/" + "test_" + name + ".png", img)
