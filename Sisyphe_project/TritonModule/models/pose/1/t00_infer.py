import numpy as np

np.set_printoptions(suppress=True)
import torch
import cv2

from p00_prep import transform_matrix_to_pose, pose_to_transform_matrix, euler_to_rotation_matrix
from p01_yolo import yolo_detect
from p02_6dof import EProPnP6DoF, LMSolver, RSLMSolver
from p03_post import Post_processing
from p05_visualize import show_pose, show_result

from t01_modules import set_model_onnx, prepare_img_4_6dof, get_camera_intrinsic, load_lm_model_info


def scale_obj(R, scale):
    R[0, 3] = R[0, 3] / scale[0]
    R[1, 3] = R[1, 3] / scale[1]
    R[2, 3] = R[2, 3] / scale[2]
    return R


def pnp_inference(img, network, pose_tcp_init, R_cam2tcp, box_center,
                  box_scale, object_size, camera_intrinsic, R_paw2tcp,
                  R_obj2est, scale_single, epropnp):

    one_meta = {
        "box_center": box_center,
        "box_scale": box_scale,
        "object_size": object_size,
        "camera_intrinsic": camera_intrinsic,
        "epropnp": epropnp,
        "out_res": 64,
        "bs": img.shape[0]
    }

    # DNN——通过RGB图像得到correspondences

    output_names = [x.name for x in network.get_outputs()]
    corr = network.run(output_names,
                       {network.get_inputs()[0].name: img.cpu().numpy()})

    # Epropnp——correspondences到姿态估计

    noc = torch.from_numpy(corr[0])
    w2d = torch.from_numpy(corr[1])

    pose_est = Post_processing(noc, w2d, one_meta)

    # 坐标转换——模型估计姿态到TCP姿态

    R_est2cam = np.vstack([pose_est[0], [0, 0, 0, 1]])
    # R_est2cam = np.array([[0.999385, -0.03410769, -0.00813938, 0.06162759],
    #                       [0.03409637, 0.99941736, -0.00152515, -0.13362148],
    #                       [0.00818665, 0.00124669, 0.99996573, 10.6159544],
    #                       [0, 0, 0, 1]])

    R_obj2est[:3, 3] = R_obj2est[:3, 3] * 0
    R_obj2cam = R_est2cam.dot(R_obj2est)

    R_tcp2world_0 = pose_to_transform_matrix(
        pose_tcp_init[:3], euler_to_rotation_matrix(pose_tcp_init[3:]))

    scale = [scale_single, scale_single, scale_single]
    R_obj2cam = scale_obj(R_obj2cam, scale=scale)
    R_obj2tcp = R_cam2tcp.dot(R_obj2cam)
    R_obj2world = R_tcp2world_0.dot(R_obj2tcp)

    R_paw2world = R_obj2world
    R_tcp2world_new = R_paw2world.dot(np.linalg.inv(R_paw2tcp))

    v_tcp_1 = transform_matrix_to_pose(R_tcp2world_new)

    # point_color = (0, 0, 255)  # BGR
    # position, position_w, center_R_w, center_t_w = show_pose(img, pose_infer[:3, :], point_color,
    #                                                          self.para_dict["object_info"][1],
    #                                                          (pose @ Rt_tcp2cam)[:3, :])
    # show_result(img, pose_infer, position, position_w, center_t_w, camera_intrinsic,
    #             name="crawl")

    return v_tcp_1


def run_pos1(img, path_onnx_yolo, pad_ratio):
    model_yolo = set_model_onnx(onnx_path=path_onnx_yolo)

    pred = yolo_detect(img,
                       model_yolo,
                       conf_threshold=0.5,
                       iou_threshold=0.45,
                       pre_treat_mode=2,
                       size_scale=3.265)

    bbox = pred.cpu().numpy()[0]
    bbox = np.array([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])

    d_h_half = (bbox[3] - bbox[1]) * pad_ratio / 2
    d_w_half = (bbox[2] - bbox[0]) * pad_ratio / 2
    d_s_half = int(max(d_h_half, d_w_half))

    h_center = int((bbox[3] + bbox[1]) / 2)
    w_center = int((bbox[2] + bbox[0]) / 2)

    img_crop = img[h_center - d_s_half:h_center + d_s_half,
                   w_center - d_s_half:w_center + d_s_half, :]

    img_4_6dof = cv2.resize(img_crop, (res_scale, res_scale))

    cv2.imwrite('img_4_6dof.png', img_4_6dof)

    box_center = torch.from_numpy(np.array([w_center, h_center]).reshape(1, 2))
    box_scale = torch.from_numpy(np.array(d_s_half * 2).reshape(1, ))

    return img_4_6dof, box_center, box_scale


def run_pos2(img_4_6dof, path_onnx_dnn, box_center, box_scale, object_info,
             camera_intrinsic, pose_tcp_init, R_cam2tcp, R_paw2tcp, R_obj2est,
             scale):

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

    pred_6dof = pnp_inference(img_4_6dof, model_dnn, pose_tcp_init, R_cam2tcp,
                              box_center, box_scale, object_size,
                              camera_intrinsic, R_paw2tcp, R_obj2est, scale,
                              model_pnp)

    return pred_6dof


if __name__ == "__main__":

    path_img = './distroted_img2.png'
    path_onnx_yolo = 'yolo.onnx'
    path_onnx_dnn = 'tmp.onnx'
    path_det_res = 'img_yolo_det.png'
    path_cad_info = 'parameter/models_info.txt'
    path_camera_intrinsic = 'parameter/camera_intrinsic.txt'
    path_R_cam2tcp = 'parameter/cam_in_tcp.txt'
    path_R_paw2tcp = 'parameter/paw_in_tcp.txt'
    path_algn = 'parameter/align_RT.txt'
    path_scale = 'parameter/scale.txt'

    scale = np.loadtxt(path_scale)
    camera_intrinsic = get_camera_intrinsic(path_camera_intrinsic)
    CAD_info = load_lm_model_info(path_cad_info)
    R_cam2tcp = np.loadtxt(path_R_cam2tcp)
    R_paw2tcp = np.loadtxt(path_R_paw2tcp)
    R_obj2est = np.loadtxt(path_algn)

    pad_ratio = 1.5
    res_scale = 256

    pose_tcp_init = np.array(
        [128.94, 319.599, 450.547, 177.998, -20.0003, -0.00060134])
    img = cv2.imread(path_img)

    img_4_6dof, box_center, box_scale = run_pos1(img, path_onnx_yolo,
                                                 pad_ratio)

    pred_6dof = run_pos2(img_4_6dof, path_onnx_dnn, box_center, box_scale,
                         CAD_info, camera_intrinsic, pose_tcp_init, R_cam2tcp,
                         R_paw2tcp, R_obj2est, scale)

    print(pred_6dof)
