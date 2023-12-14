import cv2
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from p02_6dof import PerspectiveCamera, AdaptiveHuberPnPCost


def get_world_pose(pose0, pose2, pose3):
    pose = pose0 @ pose2 @ pose3
    return pose

def Post_processing(noc, w2d, one_meta):
    object_size = one_meta["object_size"]
    box_center = one_meta["box_center"]
    box_scale = one_meta["box_scale"]
    out_res = one_meta["out_res"]
    epropnp = one_meta["epropnp"]
    bs = one_meta["bs"]
    camera_intrinsic = one_meta["camera_intrinsic"]

    dim = object_size
    dim = noc.new_tensor(dim)  # (n, 3)
    # print("dim.shape:::", dim.shape)
    # building 2D-3D correspondences
    x3d = noc.permute(0, 2, 3, 1) * dim[:, None, None, :]
    w2d = w2d.permute(0, 2, 3, 1)  # (n, h, w, 2)

    c_box = box_center.to(noc.device)
    s_box = box_scale.to(noc.device)

    s = s_box.to(torch.int64)  # (n, )

    wh_begin = c_box.to(torch.int64) - s[:, None] / 2.  # (n, 2)
    wh_unit = s.to(torch.float32) / out_res  # (n, )

    wh_arange = torch.arange(out_res, device=x3d.device, dtype=torch.float32)
    y, x = torch.meshgrid(wh_arange, wh_arange)  # (h, w)

    x2d = torch.stack(
        (wh_begin[:, 0, None, None] + x * wh_unit[:, None, None],
         wh_begin[:, 1, None, None] + y * wh_unit[:, None, None]),
        dim=-1)

    dist_coeffs = np.zeros((4, 1),
                           dtype=np.float32)  # Assuming no lens distortion

    # for fair comparison we use EPnP initialization
    pred_conf_np = w2d.mean(dim=-1).detach().cpu().numpy()  # (n, h, w)
    binary_mask = pred_conf_np >= np.quantile(pred_conf_np.reshape(bs, -1),
                                              0.8,
                                              axis=1,
                                              keepdims=True)[..., None]

    R_quats = []
    T_vectors = []
    x2d_np = x2d.detach().cpu().numpy()
    x3d_np = x3d.detach().cpu().numpy()

    for x2d_np_, x3d_np_, mask_np_ in zip(x2d_np, x3d_np, binary_mask):
        _, R_vector, T_vector = cv2.solvePnP(x3d_np_[mask_np_],
                                             x2d_np_[mask_np_],
                                             camera_intrinsic,
                                             dist_coeffs,
                                             flags=cv2.SOLVEPNP_EPNP)
        q = R.from_rotvec(R_vector.reshape(-1)).as_quat()[[3, 0, 1, 2]]
        R_quats.append(q)
        T_vectors.append(T_vector.reshape(-1))

    R_quats = x2d.new_tensor(R_quats)
    T_vectors = x2d.new_tensor(T_vectors)
    pose_init = torch.cat((T_vectors, R_quats), dim=-1)  # (n, 7)

    # Gauss-Newton optimize
    x2d = x2d.reshape(bs, -1, 2)
    w2d = w2d.reshape(bs, -1, 2)
    x3d = x3d.reshape(bs, -1, 3)
    cam_intrinsic = torch.from_numpy(camera_intrinsic).to(noc.device)
    camera = PerspectiveCamera(cam_mats=cam_intrinsic[None].expand(bs, -1, -1),
                               z_min=0.01)
    cost_fun = AdaptiveHuberPnPCost(relative_delta=0.1)

    cost_fun.set_param(x2d, w2d)
    pose_opt = epropnp(x3d,
                       x2d,
                       w2d,
                       camera,
                       cost_fun,
                       pose_init=pose_init,
                       fast_mode=True)[0]

    if torch.isnan(pose_opt).any():
        pose_est = np.zeros((bs, 3, 4), dtype=int) + 10
    else:
        T_vectors, R_quats = pose_opt.split([3, 4], dim=-1)  # (n, [3, 4])
        R_matrix = R.from_quat(
            R_quats[:, [1, 2, 3, 0]].cpu().numpy()).as_matrix()  # (n, 3, 3)
        pose_est = np.concatenate(
            [R_matrix, T_vectors.reshape(bs, 3, 1).cpu().numpy()], axis=-1)

    return pose_est
