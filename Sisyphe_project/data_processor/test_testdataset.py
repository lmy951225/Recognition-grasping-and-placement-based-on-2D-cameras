import numpy as np

np.set_printoptions(suppress=True)
import torch
import cv2
import json
import os
from pandas import DataFrame 
import time

from p00_prep import transform_matrix_to_pose, pose_to_transform_matrix, euler_to_rotation_matrix
# from p01_yolo import yolo_detect
from p02_6dof import EProPnP6DoF, LMSolver, RSLMSolver
from p03_post import Post_processing
from p05_visualize import show_pose

from t01_modules import set_model_onnx, prepare_img_4_6dof, get_camera_intrinsic, load_lm_model_info


def read_json_file(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as json_fr:
        json_data = json.load(json_fr)
        for idx, point in enumerate(json_data['shapes']):
            shape_type = point['shape_type']
            if shape_type  == 'rectangle':
                x1, y1 = point['points'][0]
                x2, y2 = point['points'][1]
                bbox = np.array([int(x1-0.5), int(y1-0.5), int(x2+0.5), int(y2+0.5)])
    return bbox


def pnp_inference(img, network, pose_tcp_init, T_cam_in_tcp, box_center,
                  box_scale, object_size, camera_intrinsic, T_jaw_in_tcp,
                  T_grasp_in_obj, epropnp):
        """Full pnp inference steps.

        :param img: img after zoom-in, for camera pose inference
        :param distorted_img: origin full image for visualize
        :param box_center: box center on full image
        :param box_scale: box scale for visualize
        :param pose_tcp_in_world: tcp pose in world or base by quat presentation
        :param guide_model: guide model for pose estimation
        :param name: graph flag, usually \'crawl\'
        :return
            poses_obj_in_world: objects 6dof in world coordinate system.
        """
        one_meta = {
            "box_center": box_center,
            "box_scale": box_scale,
            "object_size": object_size,
            "camera_intrinsic": camera_intrinsic,
            "epropnp": epropnp,
            "out_res": 64,
            "bs": img.shape[0]
        }
        output_names = [x.name for x in network.get_outputs()]
       
        corr = network.run(output_names,
                       {network.get_inputs()[0].name: img.cpu().numpy()})

       # Epropnp——correspondences到姿态估计

        noc = torch.from_numpy(corr[0])
        w2d = torch.from_numpy(corr[1])
        # object poses in cam, pose^{camera}_{object}
        pose_est = Post_processing(noc, w2d, one_meta)
        # convert scale to cad
        pose_est[:, :3, -1] *= 1000 ######
        T_obj_in_cam = np.vstack([pose_est[0], [0, 0, 0, 1]])
        T_grasp_in_cam = T_obj_in_cam.dot(T_grasp_in_obj)

        T_tcp_in_w = pose_to_transform_matrix(
            pose_tcp_init[:3], euler_to_rotation_matrix(pose_tcp_init[3:]))

        T_grasp_in_tcp = T_cam_in_tcp.dot(T_grasp_in_cam)
        T_grasp_in_w = T_tcp_in_w.dot(T_grasp_in_tcp)

        T_jaw_in_w = T_grasp_in_w
        T_tcp_in_w_new = T_jaw_in_w.dot(np.linalg.inv(T_jaw_in_tcp))

        v_tcp_1 = transform_matrix_to_pose(T_tcp_in_w_new)
        

        return v_tcp_1, T_obj_in_cam

def get_camera_intrinsic(path_camera_intrinsic):

    with open(path_camera_intrinsic) as fr:
        camera_infos = fr.readlines()[3].split()
        if camera_infos[1] == "SIMPLE_RADIAL":
            fx = float(camera_infos[4])
            fy = float(camera_infos[4])
            cx = float(camera_infos[5])
            cy = float(camera_infos[6])

        else:
            fx = float(camera_infos[4])
            fy = float(camera_infos[5])
            cx = float(camera_infos[6])
            cy = float(camera_infos[7])

        camera_intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]],
                                    dtype=np.float32)

    return camera_intrinsic


def compute_pose(img, bbox, pad_ratio, path_onnx_dnn, object_info,
             camera_intrinsic, pose_tcp_init, T_cam_in_tcp, T_jaw_in_tcp, T_grasp_in_obj):

    d_h_half = (bbox[3] - bbox[1]) * pad_ratio / 2
    d_w_half = (bbox[2] - bbox[0]) * pad_ratio / 2
    d_s_half = int(max(d_h_half, d_w_half))

    h_half = int((bbox[3] - bbox[1])  / 2)
    w_half = int((bbox[2] - bbox[0])  / 2)

    h_center = int((bbox[3] + bbox[1]) / 2)
    w_center = int((bbox[2] + bbox[0]) / 2)
    img_crop = np.zeros((d_s_half*2, d_s_half*2, 3))

    img_crop[d_s_half - h_half:d_s_half + h_half, d_s_half - w_half:d_s_half + w_half] = img[bbox[1]:(2*h_half+bbox[1]), bbox[0]:(2*w_half+bbox[0])]
    img_4_6dof = cv2.resize(img_crop, (res_scale, res_scale))
    cv2.imwrite('img_4_6dof.png', img_4_6dof)

    box_center = torch.from_numpy(np.array([w_center, h_center]).reshape(1, 2))
    box_scale = torch.from_numpy(np.array(d_s_half * 2).reshape(1, ))

    object_size = torch.from_numpy(
        np.asarray([[
            abs(object_info[1]['min_x']),
            abs(object_info[1]['min_y']),
            abs(object_info[1]['min_z'])
        ]]))

    model_dnn = set_model_onnx(onnx_path=path_onnx_dnn)

    model_pnp = EProPnP6DoF(mc_samples=512,
                            num_iter=4,
                            solver=LMSolver(dof=6,
                                            num_iter=5,
                                            init_solver=RSLMSolver(
                                                dof=6,
                                                num_points=16,
                                                num_proposals=4,
                                                num_iter=3)))

    img_4_6dof = prepare_img_4_6dof(img_4_6dof)

    pred_6dof, T_obj_in_cam = pnp_inference(img_4_6dof, model_dnn, pose_tcp_init, T_cam_in_tcp,
                              box_center, box_scale, object_size,
                              camera_intrinsic, T_jaw_in_tcp, T_grasp_in_obj,
                              model_pnp)
    T_tcp_in_w = pose_to_transform_matrix(
            pose_tcp_init[:3], euler_to_rotation_matrix(pose_tcp_init[3:]))
    position, _, _, _ = show_pose(T_obj_in_cam[:3, :], object_info[1],
                    (T_tcp_in_w @ T_cam_in_tcp))
    show_box(img, T_obj_in_cam, position, camera_intrinsic)


    return pred_6dof

def show_box(img, pose, position, cam_k):
    result = "result"
    if not os.path.exists(result):
        os.makedirs(result)

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
    cv2.imwrite("result/" + "test" +str(time.time()) +  ".png", img)



if __name__ == "__main__":

    path_img = '/media/jingjing/56a1e372-b7ba-43a2-8adc-ce445bc31fff/jingjing/share/data/bydGT_1101/test/images_flip_undis/'
    path_onnx_dnn = 'new_server_11_03_chr/tmp_sphere.onnx'
    path_cad_info = 'models_info.txt'
    path_camera_intrinsic = 'camera_intrinsic.txt'

    img_files = sorted(os.listdir(path_img), key = lambda s: int((s.split('.')[0]).split('-')[1]))
    camera_intrinsic = get_camera_intrinsic(path_camera_intrinsic)
    CAD_info = load_lm_model_info(path_cad_info)

    T_cam_in_tcp = np.array([[ -0.01985869  , 0.99978282  , 0.00632089 , 56.68305607],
                        [ -0.99972596 , -0.01993514 ,  0.01227191 ,  0.50922745],
                        [  0.01239525 , -0.00607545  , 0.99990472 ,190.8758467 ],
                        [  0.     ,      0.   ,        0.    ,       1.        ]])

    T_jaw_in_tcp = np.array([
                         [1,0,0,-0.77648664],
                         [0,-1,0,1.41247193],
                         [0,0,-1,238.30122821],
                         [0,0,0,1]
                        ])

    T_grasp_in_obj = np.array([[1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, -6.423000496954500704e-02],
                             [0.000000000000000000e+00, 9.510565162951535312e-01, 3.090169943749473958e-01, 1.033519952625903571e+01],
                             [0.000000000000000000e+00, -3.090169943749473958e-01, 9.510565162951535312e-01, 1.371617754691659918e+01],
                             [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
    pad_ratio = 1.5
    res_scale = 256

    pose_tcp_init = np.array(
        [128.94, 319.599, 450.547, 177.998, -20.0003, -0.000601335])
    
    result = []
    name = []
    for file in img_files[0:]:
        if file.endswith('.jpg'):
            img = cv2.imread(path_img + file)
            bbox = read_json_file(path_img + file.replace('jpg', 'json'))
            pred_6dof = compute_pose(img, bbox, pad_ratio, path_onnx_dnn,
                                CAD_info, camera_intrinsic, pose_tcp_init, T_cam_in_tcp,
                                T_jaw_in_tcp, T_grasp_in_obj)
            print(file, pred_6dof)
            result.append(pred_6dof)
            name.append(file)
    results = np.array(result)
    position = ['(%s,%s)' %(i,j) for i in range(3) for j in range(5) for k in range(12)]
    angle = [i for i in [0, 90, 180, -90] for j in range(3)] * 15
    bg = ['black', 'white', 'chess'] * 60
    df = DataFrame(data = {'name': name, 'Tx':results[:,0], 'Ty':results[:,1], 'Tz':results[:,2], 
                           'Rx':results[:,3], 'Ry':results[:,4], 'Rz':results[:,5], 
                           'position': position, 'angle': angle, 'bg': bg})
    
    df.to_csv('test.csv')

