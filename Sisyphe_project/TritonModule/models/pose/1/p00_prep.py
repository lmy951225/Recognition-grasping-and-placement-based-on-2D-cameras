import os
import numpy as np
import cv2
import json
import torch
import math

from mmpose.apis import init_model
from scipy.spatial.transform import Rotation as R


def load_lm_model_info(info_pth):
    """load object info from models_info.txt.

    :param info_pth: txt file path
    :return
        infos: real object info.
    """
    infos = {}
    with open(info_pth, 'r') as f:
        for line in f.readlines():
            items = line.strip().split(' ')

            cls_idx = int(items[0])
            infos[cls_idx] = {}
            infos[cls_idx]['diameter'] = float(
                items[2]) / 1000.  # unit: mm => m
            infos[cls_idx]['min_x'] = float(items[4]) / 1000.
            infos[cls_idx]['min_y'] = float(items[6]) / 1000.
            infos[cls_idx]['min_z'] = float(items[8]) / 1000.
            infos[cls_idx]['max_x'] = (float(items[10]) +
                                       float(items[4])) / 1000.
            infos[cls_idx]['max_y'] = (float(items[12]) +
                                       float(items[6])) / 1000.
            infos[cls_idx]['max_z'] = (float(items[14]) +
                                       float(items[8])) / 1000.
            infos[cls_idx]['center_x'] = float(items[16]) / 1000.
            infos[cls_idx]['center_y'] = float(items[18]) / 1000.
            infos[cls_idx]['center_z'] = float(items[20]) / 1000.
    return infos

def get_parameter(para_path="./parameter"):
    """Get parameters for inference.

    :param para_path: the directory which all config files in.
    :return
        para_dict: config parameters.
    """

    para_dict = {}
    camera_intrinsic_file = os.path.join(para_path, "camera_intrinsic.txt")
    T_grasp_in_obj = np.loadtxt(os.path.join(para_path, 'T_grasp_in_obj.txt'))
    cam_in_tcp_path = os.path.join(para_path, "cam_in_tcp.txt")
    align_RT_path = os.path.join(para_path, "align_RT.txt")
    radius_path = os.path.join(para_path, "radius.txt")
    paw_in_tcp_path = os.path.join(para_path, "paw_in_tcp.txt")
    object_size_path = os.path.join(para_path, "models_info.txt")
    kp = json.load(
        open(os.path.join(para_path, "undistort_template.json"), 'r'))
    radius = (np.loadtxt(os.path.join(radius_path)))
    pts_2d_config = os.path.join(para_path, "hrnet.py")
    pts_2d_model = os.path.join(para_path, "epoch_120.pth")

    # from grasp to obj center
    para_dict["T_grasp_in_obj"] = T_grasp_in_obj

    # 2d pts config
    para_dict["pts_2d_config"] = pts_2d_config
    para_dict["pts_2d_model"] = pts_2d_model

    para_dict["select_r"] = radius[0]
    para_dict["guide_r"] = radius[1]

    cam_in_tcp = np.loadtxt(cam_in_tcp_path)
    if cam_in_tcp.shape[0] * cam_in_tcp.shape[1] == 16:
        cam_in_tcp = cam_in_tcp
    else:
        cam_in_tcp = np.vstack([cam_in_tcp, [0, 0, 0, 1]])

    para_dict["cam_in_tcp"] = cam_in_tcp

    align_RT = np.loadtxt(align_RT_path)
    if align_RT.shape[0] * align_RT.shape[1] == 16:
        align_RT = align_RT
    else:
        align_RT = np.vstack([align_RT, [0, 0, 0, 1]])

    paw_in_tcp = np.loadtxt(paw_in_tcp_path)
    if paw_in_tcp.shape[0] * paw_in_tcp.shape[1] == 16:
        paw_in_tcp = paw_in_tcp
    else:
        paw_in_tcp = np.vstack([paw_in_tcp, [0, 0, 0, 1]])

    para_dict["paw_in_tcp"] = paw_in_tcp

    object_info = load_lm_model_info(object_size_path)
    object_size = torch.from_numpy(
        np.asarray([[
            abs(object_info[1]['min_x']),
            abs(object_info[1]['min_y']),
            abs(object_info[1]['min_z'])
        ]]))

    para_dict["object_info"] = object_info
    para_dict["object_size"] = object_size
    pose_model = init_model(para_dict["pts_2d_config"],
                            para_dict["pts_2d_model"],
                            cfg_options=None)
    pose_model = pose_model.cuda()
    para_dict["pose_model"] = pose_model

    camera_intrinsic_file = os.path.join(para_path, "camera_intrinsic.txt")
    with open(camera_intrinsic_file) as fr:
        camera_infos = fr.readlines()[3].split()
        if camera_infos[1] == "SIMPLE_RADIAL":
            fx = float(camera_infos[4])
            fy = float(camera_infos[4])
            cx = float(camera_infos[5])
            cy = float(camera_infos[6])
            width = int(camera_infos[2])
            height = int(camera_infos[3])
            distortion = float(camera_infos[7])
        else:
            fx = float(camera_infos[4])
            fy = float(camera_infos[5])
            cx = float(camera_infos[6])
            cy = float(camera_infos[7])
            width = int(camera_infos[2])
            height = int(camera_infos[3])
            distortion = np.array([float(camera_infos[idx]) for idx in range(8,13)])
        camera_intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]],
                                    dtype=np.float32)

    para_dict["camera_intrinsic"] = camera_intrinsic
    para_dict["width"] = width
    para_dict["height"] = height
    para_dict["distortion"] = distortion
    para_dict["kp"] = kp

    return para_dict

# ----------------------------------------------------------------------------------------------------------------------
# RGB data
# ----------------------------------------------------------------------------------------------------------------------

def zoom_in(im, c, s, res, channel=3, interpolate=cv2.INTER_LINEAR):
    """
    Zoom in on the object with center c and size s, and resize to resolution res.
    :param im: nd.array, single-channel or 3-channel image
    :param c: (w, h), object center
    :param s: scalar, object size
    :param res: target resolution
    :param channel: image channels.
    :param interpolate: cv2 parameter.
    :return
        im_resize: zoomed object patch.
        c_h: new height center.
        c_s: new width center.
        s: scale
    """
    c_w, c_h = c
    c_w, c_h, s, res = int(c_w), int(c_h), int(s), int(res)
    if channel == 1:
        im = im[..., None]
    h, w = im.shape[:2]
    u = int(c_h - 0.5 * s + 0.5)
    l = int(c_w - 0.5 * s + 0.5)
    b = u + s
    r = l + s
    if (u >= h) or (l >= w) or (b <= 0) or (r <= 0):
        return np.zeros((res, res, channel)).squeeze()
    if u < 0:
        local_u = -u
        u = 0
    else:
        local_u = 0
    if l < 0:
        local_l = -l
        l = 0
    else:
        local_l = 0
    if b > h:
        local_b = s - (b - h)
        b = h
    else:
        local_b = s
    if r > w:
        local_r = s - (r - w)
    else:
        local_r = s
    im_crop = np.zeros((s, s, channel))
    im_crop[local_u:local_b, local_l:local_r, :] = im[u:b, l:r, :]
    im_crop = im_crop.squeeze()
    im_resize = cv2.resize(im_crop, (res, res), interpolation=interpolate)
    c_h = 0.5 * (u + b)
    c_w = 0.5 * (l + r)
    s = s
    return im_resize, c_h, c_w, s

def xyxy_to_cs(bbox_xyxy, s_ratio, s_max=None):
    """convert xyxy format to center-side format.

    :param bbox_xyxy: tl and br xyxy format.
    :param s_ratio: ratios.
    :param s_max: limit max side value.
    :return
        c: center point
        s: max side
    """
    x_min, y_min, x_max, y_max = bbox_xyxy
    w = int(x_max) - int(x_min)
    h = int(y_max) - int(y_min)

    c = np.array((x_min + 0.5 * w, y_min + 0.5 * h))  # [c_w, c_h]
    s = max(w, h) * s_ratio
    if s_max != None:
        s = min(s, s_max)
    return c, s

def undistort(img, k, d, width, height):
    """Undistort the image.

    :param frame: Input image.
    :param k: Camera intrinsic params.
    :param d: Distortion params.
    :param width: Image width.
    :param height: Image height.
    """
    h0, w0 = img.shape[0:2]
    # Warning: (h,w) must be same as camera capture
    # h, w = (1104, 2092)
    h = height
    w = width
    undistorted_image = cv2.undistort(img, k, d)
    result = np.full((h, w, 3), (0,0,0), dtype=np.uint8)
    # compute center offset
    x_center = (w - w0) // 2
    y_center = (h - h0) // 2
    # copy img image into center of result image
    result[y_center:y_center+h0,
        x_center:x_center+w0] = undistorted_image
    return result

# ----------------------------------------------------------------------------------------------------------------------
# matrix convert
# ----------------------------------------------------------------------------------------------------------------------

def skew(x):
    """
    Args:
        x (torch.Tensor): shape (*, 3)

    Returns:
        torch.Tensor: (*, 3, 3), skew symmetric matrices
    """
    mat = x.new_zeros(x.shape[:-1] + (3, 3))
    mat[..., [2, 0, 1], [1, 2, 0]] = x
    mat[..., [1, 2, 0], [2, 0, 1]] = -x
    return mat

def quaternion_to_rot_mat(quaternions):
    """
    Args:
        quaternions (torch.Tensor): (*, 4)

    Returns:
        torch.Tensor: (*, 3, 3)
    """
    if quaternions.requires_grad:
        w, i, j, k = torch.unbind(quaternions, -1)
        rot_mats = torch.stack(
            (1 - 2 * (j * j + k * k), 2 * (i * j - k * w), 2 *
             (i * k + j * w), 2 * (i * j + k * w), 1 - 2 * (i * i + k * k), 2 *
             (j * k - i * w), 2 * (i * k - j * w), 2 * (j * k + i * w), 1 - 2 *
             (i * i + j * j)),
            dim=-1,
        ).reshape(quaternions.shape[:-1] + (3, 3))
    else:
        w, v = quaternions.split([1, 3], dim=-1)
        rot_mats = 2 * (w.unsqueeze(-1) * skew(v) +
                        v.unsqueeze(-1) * v.unsqueeze(-2))
        diag = torch.diagonal(rot_mats, dim1=-2, dim2=-1)
        diag += w * w - (v.unsqueeze(-2) @ v.unsqueeze(-1)).squeeze(-1)
    return rot_mats

def xyz_rxryrz2transformation(xyz_rxryrz):  # only work for fannuc
    """xyz to rxryrz2transformation.

    :param xyz_rxryrz: (1,6) array, contains (x, y, z, yaw, pitch, roll)
    :return
        transformation: a 4x4 RT matrix
    """
    transformation = np.identity(4)
    transformation[:3, :3] = R.from_euler(seq="xyz",
                                          angles=xyz_rxryrz[3:],
                                          degrees=True).as_matrix()
    transformation[0, 3] = xyz_rxryrz[0]
    transformation[1, 3] = xyz_rxryrz[1]
    transformation[2, 3] = xyz_rxryrz[2]
    return transformation

def pose_to_transform_matrix(translation, rotation):
    transform_matrix = np.eye(4)
    transform_matrix[:3, 3] = translation
    transform_matrix[:3, :3] = rotation

    return transform_matrix

def euler_to_rotation_matrix(euler_angle):
    roll, pitch, yaw = euler_angle
    # convert euler angles to radians
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    # compute rotation matrix 
    cos_roll = math.cos(roll_rad)
    sin_roll = math.sin(roll_rad)
    cos_pitch = math.cos(pitch_rad)
    sin_pitch = math.sin(pitch_rad)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    rotation_matrix = np.array(
        [[
            cos_yaw * cos_pitch,
            cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
            cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
        ],
         [
             sin_yaw * cos_pitch,
             sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
             sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
         ],
         [
             -sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll
        ]])

    return rotation_matrix

def transform_matrix_to_pose(transform_matrix):
    translation = transform_matrix[:3, 3]
    rotation = transform_matrix[:3, :3]

    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    pitch = np.arctan2(-rotation[2, 0],
                       np.sqrt(rotation[2, 1]**2 + rotation[2, 2]**2))
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])

    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    pose = np.array(
        [translation[0], translation[1], translation[2], roll, pitch, yaw])

    return pose
