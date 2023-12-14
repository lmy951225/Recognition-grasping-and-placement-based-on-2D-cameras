import open3d as o3d
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta
from functools import partial

from torch.distributions.multivariate_normal import _batch_mahalanobis, _standard_normal, _batch_mv
from pyro.distributions import TorchDistribution, constraints
from pyro.distributions.util import broadcast_shape

from p00_prep import skew, quaternion_to_rot_mat


def evaluate_pnp(x3d,
                 x2d,
                 w2d,
                 pose,
                 camera,
                 cost_fun,
                 out_jacobian=False,
                 out_residual=False,
                 out_cost=False,
                 **kwargs):
    """
    Args:
        x3d (torch.Tensor): Shape (*, n, 3)
        x2d (torch.Tensor): Shape (*, n, 2)
        w2d (torch.Tensor): Shape (*, n, 2)
        pose (torch.Tensor): Shape (*, 4 or 7)
        camera: Camera object of batch size (*, )
        cost_fun: PnPCost object of batch size (*, )
        out_jacobian (torch.Tensor | bool): When a tensor is passed, treated as the output tensor;
            when True, returns the Jacobian; when False, skip the computation and returns None
        out_residual (torch.Tensor | bool): When a tensor is passed, treated as the output tensor;
            when True, returns the residual; when False, skip the computation and returns None
        out_cost (torch.Tensor | bool): When a tensor is passed, treated as the output tensor;
            when True, returns the cost; when False, skip the computation and returns None

    Returns:
        Tuple:
            residual (torch.Tensor | None): Shape (*, n*2)
            cost (torch.Tensor | None): Shape (*, )
            jacobian (torch.Tensor | None): Shape (*, n*2, 4 or 6)
    """
    x2d_proj, jac_cam = camera.project(
        x3d,
        pose,
        out_jac=(out_jacobian.view(x2d.shape[:-1] + (2, out_jacobian.size(-1)))
                 if isinstance(out_jacobian, torch.Tensor) else out_jacobian),
        **kwargs)
    residual, cost, jacobian = cost_fun.compute(x2d_proj,
                                                x2d,
                                                w2d,
                                                jac_cam=jac_cam,
                                                out_residual=out_residual,
                                                out_cost=out_cost,
                                                out_jacobian=out_jacobian)
    return residual, cost, jacobian

def pnp_normalize(x3d, pose=None, detach_transformation=True):
    """
    Args:
        x3d (torch.Tensor): Shape (*, n, 3)
        pose (torch.Tensor | None): Shape (*, 4)
        detach_transformation (bool)

    Returns:
        Tuple[torch.Tensor]:
            offset: Shape (*, 1, 3)
            x3d_norm: Shape (*, n, 3), normalized x3d
            pose_norm: Shape (*, ), transformed pose
    """
    offset = torch.mean(x3d.detach() if detach_transformation else x3d,
                        dim=-2)  # (*, 3)
    x3d_norm = x3d - offset.unsqueeze(-2)
    if pose is not None:
        pose_norm = torch.empty_like(pose)
        pose_norm[..., 3:] = pose[..., 3:]
        pose_norm[..., :3] = pose[..., :3] + \
            ((yaw_to_rot_mat(pose[..., 3]) if pose.size(-1) == 4
              else quaternion_to_rot_mat(pose[..., 3:])) @ offset.unsqueeze(-1)).squeeze(-1)
    else:
        pose_norm = None
    return offset, x3d_norm, pose_norm

def pnp_denormalize(offset, pose_norm):
    pose = torch.empty_like(pose_norm)
    pose[..., 3:] = pose_norm[..., 3:]
    pose[..., :3] = pose_norm[..., :3] - \
        ((yaw_to_rot_mat(pose_norm[..., 3]) if pose_norm.size(-1) == 4
          else quaternion_to_rot_mat(pose_norm[..., 3:])) @ offset.unsqueeze(-1)).squeeze(-1)
    return pose

def solve_wrapper(b, A):
    if A.numel() > 0:
        return torch.linalg.solve(A, b)
    else:
        return b + A.reshape_as(b)

def project_a(x3d, pose, cam_mats, z_min: float):
    if pose.size(-1) == 4:
        x3d_rot = x3d @ (yaw_to_rot_mat(pose[..., -1])).transpose(-1, -2)
    else:
        x3d_rot = x3d @ quaternion_to_rot_mat(pose[..., 3:]).transpose(-1, -2)
    x2dh_proj = (x3d_rot + pose[..., None, :3]) @ cam_mats.transpose(-1, -2)
    z = x2dh_proj[..., 2:3].clamp(min=z_min)
    x2d_proj = x2dh_proj[..., :2] / z  # (*, n, 2)
    return x2d_proj, x3d_rot, z

def project_b(x3d, pose, cam_mats, z_min: float):
    if pose.size(-1) == 4:
        x2dh_proj = x3d @ (cam_mats @ yaw_to_rot_mat(pose[..., -1])).transpose(-1, -2) \
                    + (cam_mats @ pose[..., :3, None]).squeeze(-1).unsqueeze(-2)
    else:
        x2dh_proj = x3d @ (cam_mats @ quaternion_to_rot_mat(pose[..., 3:])).transpose(-1, -2) \
                    + (cam_mats @ pose[..., :3, None]).squeeze(-1).unsqueeze(-2)
    z = x2dh_proj[..., 2:3].clamp(min=z_min)
    x2d_proj = x2dh_proj[..., :2] / z
    return x2d_proj, z

def huber_kernel(s_sqrt, delta):
    half_rho = torch.where(delta >= s_sqrt, 0.5 * torch.square(s_sqrt),
                           delta * s_sqrt - 0.5 * torch.square(delta))
    return half_rho

def huber_d_kernel(s_sqrt, delta, eps: float = 1e-10):
    if s_sqrt.requires_grad or delta.requires_grad:
        rho_d_sqrt = (delta.clamp(min=eps).sqrt() *
                      s_sqrt.clamp(min=eps).rsqrt()).clamp(max=1.0)
    else:
        rho_d_sqrt = (delta / s_sqrt.clamp_(min=eps)).clamp_(max=1.0).sqrt_()
    return rho_d_sqrt

def yaw_to_rot_mat(yaw):
    """
    Args:
        yaw (torch.Tensor): (*)

    Returns:
        torch.Tensor: (*, 3, 3)
    """
    sin_yaw = torch.sin(yaw)
    cos_yaw = torch.cos(yaw)

    rot_mats = yaw.new_zeros(yaw.shape + (3, 3))
    rot_mats[..., 0, 0] = cos_yaw
    rot_mats[..., 2, 2] = cos_yaw
    rot_mats[..., 0, 2] = sin_yaw
    rot_mats[..., 2, 0] = -sin_yaw
    rot_mats[..., 1, 1] = 1
    return rot_mats

class HuberPnPCost(object):

    def __init__(self, delta=1.0, eps=1e-10):
        super(HuberPnPCost, self).__init__()
        self.eps = eps
        self.delta = delta

    def set_param(self, *args, **kwargs):
        pass

    def compute(self,
                x2d_proj,
                x2d,
                w2d,
                jac_cam=None,
                out_residual=False,
                out_cost=False,
                out_jacobian=False):
        """
        Args:
            x2d_proj: Shape (*, n, 2)
            x2d: Shape (*, n, 2)
            w2d: Shape (*, n, 2)
            jac_cam: Shape (*, n, 2, 4 or 6), Jacobian of x2d_proj w.r.t. pose
            out_residual (Tensor | bool): Shape (*, n*2) or equivalent shape
            out_cost (Tensor | bool): Shape (*, )
            out_jacobian (Tensor | bool): Shape (*, n*2, 4 or 6) or equivalent shape
        """
        bs = x2d_proj.shape[:-2]
        pn = x2d_proj.size(-2)
        delta = self.delta
        if not isinstance(delta, torch.Tensor):
            delta = x2d.new_tensor(delta)
        delta = delta[..., None]

        residual = (x2d_proj - x2d) * w2d
        s_sqrt = residual.norm(dim=-1)

        if out_cost is not False:
            half_rho = huber_kernel(s_sqrt, delta)
            if not isinstance(out_cost, torch.Tensor):
                out_cost = None
            cost = torch.sum(half_rho, dim=-1, out=out_cost)
        else:
            cost = None

        # robust rescaling
        if out_residual is not False or out_jacobian is not False:
            rho_d_sqrt = huber_d_kernel(s_sqrt, delta, eps=self.eps)
            if out_residual is not False:
                if isinstance(out_residual, torch.Tensor):
                    out_residual = out_residual.view(*bs, pn, 2)
                else:
                    out_residual = None
                residual = torch.mul(residual,
                                     rho_d_sqrt[..., None],
                                     out=out_residual).view(*bs, pn * 2)
            if out_jacobian is not False:
                assert jac_cam is not None
                dof = jac_cam.size(-1)
                if isinstance(out_jacobian, torch.Tensor):
                    out_jacobian = out_jacobian.view(*bs, pn, 2, dof)
                else:
                    out_jacobian = None
                # rescaled jacobian
                jacobian = torch.mul(jac_cam,
                                     (w2d * rho_d_sqrt[..., None])[..., None],
                                     out=out_jacobian).view(*bs, pn * 2, dof)
        if out_residual is False:
            residual = None
        if out_jacobian is False:
            jacobian = None
        return residual, cost, jacobian

    def reshape_(self, *batch_shape):
        if isinstance(self.delta, torch.Tensor):
            self.delta = self.delta.reshape(*batch_shape)
        return self

    def expand_(self, *batch_shape):
        if isinstance(self.delta, torch.Tensor):
            self.delta = self.delta.expand(*batch_shape)
        return self

    def repeat_(self, *batch_repeat):
        if isinstance(self.delta, torch.Tensor):
            self.delta = self.delta.repeat(*batch_repeat)
        return self

    def shallow_copy(self):
        return HuberPnPCost(delta=self.delta, eps=self.eps)


class AdaptiveHuberPnPCost(HuberPnPCost):

    def __init__(self, delta=None, relative_delta=0.5, eps=1e-10):
        super(HuberPnPCost, self).__init__()
        self.delta = delta
        self.relative_delta = relative_delta
        self.eps = eps

    def set_param(self, x2d, w2d):
        # compute dynamic delta
        x2d_std = torch.var(x2d, dim=-2).sum(dim=-1).sqrt()  # (num_obj, )
        self.delta = w2d.mean(
            dim=(-2, -1)) * x2d_std * self.relative_delta  # (num_obj, )

    def shallow_copy(self):
        return AdaptiveHuberPnPCost(delta=self.delta,
                                    relative_delta=self.relative_delta,
                                    eps=self.eps)


class PerspectiveCamera(object):

    def __init__(self,
                 cam_mats=None,
                 z_min=0.1,
                 img_shape=None,
                 allowed_border=200,
                 lb=None,
                 ub=None):
        """
        Args:
            cam_mats (Tensor): Shape (*, 3, 3)
            img_shape (Tensor | None): Shape (*, 2) in [h, w]
            lb (Tensor | None): Shape (*, 2), lower bound in [x, y]
            ub (Tensor | None): Shape (*, 2), upper bound in [x, y]
        """
        super(PerspectiveCamera, self).__init__()
        self.z_min = float(z_min)
        self.allowed_border = allowed_border
        self.set_param(cam_mats, img_shape, lb, ub)

    def set_param(self, cam_mats, img_shape=None, lb=None, ub=None):
        self.cam_mats = cam_mats
        if img_shape is not None:
            self.lb = -0.5 - self.allowed_border
            self.ub = img_shape[..., [1, 0]] + (-0.5 + self.allowed_border)
        else:
            self.lb = lb
            self.ub = ub

    def project(self, x3d, pose, out_jac=False, clip_jac=True):
        """
        Args:
            x3d (Tensor): Shape (*, n, 3)
            pose (Tensor): Shape (*, 4 or 7)
            out_jac (bool | Tensor): Shape (*, n, 2, 4 or 6)

        Returns:
            Tuple[Tensor]:
                x2d_proj: Shape (*, n, 2)
                jac: Shape (*, n, 2, 4 or 6), Jacobian w.r.t. the local pose in tangent space
        """
        if out_jac is not False:
            x2d_proj, x3d_rot, zcam = project_a(x3d, pose, self.cam_mats,
                                                self.z_min)
        else:
            x2d_proj, zcam = project_b(x3d, pose, self.cam_mats, self.z_min)

        lb, ub = self.lb, self.ub
        if lb is not None and ub is not None:
            requires_grad = x2d_proj.requires_grad
            if isinstance(lb, torch.Tensor):
                lb = lb.unsqueeze(-2)
                x2d_proj = torch.max(
                    lb, x2d_proj, out=x2d_proj if not requires_grad else None)
            else:
                x2d_proj.clamp_(min=lb)
            if isinstance(ub, torch.Tensor):
                ub = ub.unsqueeze(-2)
                x2d_proj = torch.min(
                    x2d_proj, ub, out=x2d_proj if not requires_grad else None)
            else:
                x2d_proj.clamp_(max=ub)

        if out_jac is not False:
            if not isinstance(out_jac, torch.Tensor):
                out_jac = None
            jac = self.project_jacobian(x3d_rot,
                                        zcam,
                                        x2d_proj,
                                        out_jac=out_jac,
                                        dof=4 if pose.size(-1) == 4 else 6)
            if clip_jac:
                if lb is not None and ub is not None:
                    clip_mask = (zcam == self.z_min) | ((x2d_proj == lb) |
                                                        (x2d_proj == ub))
                else:
                    clip_mask = zcam == self.z_min
                jac.masked_fill_(clip_mask[..., None], 0)
        else:
            jac = None

        return x2d_proj, jac

    def project_jacobian(self, x3d_rot, zcam, x2d_proj, out_jac, dof):
        if dof == 4:
            d_xzcam_d_yaw = torch.stack((x3d_rot[..., 2], -x3d_rot[..., 0]),
                                        dim=-1).unsqueeze(-1)
        elif dof == 6:
            d_x3dcam_d_rot = skew(x3d_rot * 2)
        else:
            raise ValueError('dof must be 4 or 6')
        if zcam.requires_grad or x2d_proj.requires_grad:
            assert out_jac is None, 'out_jac is not supported for backward'
            d_x2d_d_x3dcam = torch.cat(
                (self.cam_mats[..., None, :2, :2] / zcam.unsqueeze(-1),
                 (self.cam_mats[..., None, :2, 2:3] - x2d_proj.unsqueeze(-1)) /
                 zcam.unsqueeze(-1)),
                dim=-1)
            # (b, n, 2, 4 or 6)
            jac = torch.cat(
                (d_x2d_d_x3dcam, d_x2d_d_x3dcam[..., ::2] @ d_xzcam_d_yaw
                 if dof == 4 else d_x2d_d_x3dcam @ d_x3dcam_d_rot),
                dim=-1)
        else:
            if out_jac is None:
                jac = torch.empty(x3d_rot.shape[:-1] + (2, dof),
                                  device=x3d_rot.device,
                                  dtype=x3d_rot.dtype)
            else:
                jac = out_jac
            # d_x2d_d_xycam (b, n, 2, 2) = (b, 1, 2, 2) / (b, n, 1, 1)
            jac[..., :2] = self.cam_mats[...,
                                         None, :2, :2] / zcam.unsqueeze(-1)
            # d_x2d_d_zcam (b, n, 2, 1) = ((b, 1, 2, 1) - (b, n, 2, 1)) / (b, n, 1, 1)
            jac[..., 2:3] = (self.cam_mats[..., None, :2, 2:3] -
                             x2d_proj.unsqueeze(-1)) / zcam.unsqueeze(-1)
            jac[..., 3:] = jac[..., ::2] @ d_xzcam_d_yaw if dof == 4 \
                else jac[..., :3] @ d_x3dcam_d_rot
        return jac

    @staticmethod
    def get_quaternion_transfrom_mat(quaternions):
        """
        Get the transformation matrix that maps the local rotation delta in 3D tangent
        space to the 4D space where the quaternion is embedded.
        Args:
            quaternions (torch.Tensor): (*, 4), the quaternion that determines the source
                tangent space

        Returns:
            torch.Tensor: (*, 4, 3)
        """
        w, i, j, k = torch.unbind(quaternions, -1)
        transfrom_mat = torch.stack((i, j, k, -w, -k, j, k, -w, -i, -j, i, -w),
                                    dim=-1)
        return transfrom_mat.reshape(quaternions.shape[:-1] + (4, 3))

    def reshape_(self, *batch_shape):
        self.cam_mats = self.cam_mats.reshape(*batch_shape, 3, 3)
        if isinstance(self.lb, torch.Tensor):
            self.lb = self.lb.reshape(*batch_shape, 2)
        if isinstance(self.ub, torch.Tensor):
            self.ub = self.ub.reshape(*batch_shape, 2)
        return self

    def expand_(self, *batch_shape):
        self.cam_mats = self.cam_mats.expand(*batch_shape, -1, -1)
        if isinstance(self.lb, torch.Tensor):
            self.lb = self.lb.expand(*batch_shape, -1)
        if isinstance(self.ub, torch.Tensor):
            self.ub = self.ub.expand(*batch_shape, -1)
        return self

    def repeat_(self, *batch_repeat):
        self.cam_mats = self.cam_mats.repeat(*batch_repeat, 1, 1)
        if isinstance(self.lb, torch.Tensor):
            self.lb = self.lb.repeat(*batch_repeat, 1)
        if isinstance(self.ub, torch.Tensor):
            self.ub = self.ub.repeat(*batch_repeat, 1)
        return self

    def shallow_copy(self):
        return PerspectiveCamera(cam_mats=self.cam_mats,
                                 z_min=self.z_min,
                                 allowed_border=self.allowed_border,
                                 lb=self.lb,
                                 ub=self.ub)


class LMSolver(nn.Module):
    """
    Levenberg-Marquardt solver, with fixed number of iterations.

    - For 4DoF case, the pose is parameterized as [x, y, z, yaw], where yaw is the
    rotation around the Y-axis in radians.
    - For 6DoF case, the pose is parameterized as [x, y, z, w, i, j, k], where
    [w, i, j, k] is the unit quaternion.
    """

    def __init__(self,
                 dof=4,
                 num_iter=10,
                 min_lm_diagonal=1e-6,
                 max_lm_diagonal=1e32,
                 min_relative_decrease=1e-3,
                 initial_trust_region_radius=30.0,
                 max_trust_region_radius=1e16,
                 eps=1e-5,
                 normalize=False,
                 init_solver=None):
        super(LMSolver, self).__init__()
        self.dof = dof
        self.num_iter = num_iter
        self.min_lm_diagonal = min_lm_diagonal
        self.max_lm_diagonal = max_lm_diagonal
        self.min_relative_decrease = min_relative_decrease
        self.initial_trust_region_radius = initial_trust_region_radius
        self.max_trust_region_radius = max_trust_region_radius
        self.eps = eps
        self.normalize = normalize
        self.init_solver = init_solver

    def forward(self,
                x3d,
                x2d,
                w2d,
                camera,
                cost_fun,
                with_pose_opt_plus=False,
                pose_init=None,
                normalize_override=None,
                **kwargs):
        if isinstance(normalize_override, bool):
            normalize = normalize_override
        else:
            normalize = self.normalize
        if normalize:
            transform, x3d, pose_init = pnp_normalize(
                x3d, pose_init, detach_transformation=True)

        pose_opt, pose_cov, cost = self.solve(x3d,
                                              x2d,
                                              w2d,
                                              camera,
                                              cost_fun,
                                              pose_init=pose_init,
                                              **kwargs)
        if with_pose_opt_plus:
            step = self.gn_step(x3d, x2d, w2d, pose_opt, camera, cost_fun)
            pose_opt_plus = self.pose_add(pose_opt, step, camera)
        else:
            pose_opt_plus = None

        if normalize:
            pose_opt = pnp_denormalize(transform, pose_opt)
            if pose_cov is not None:
                raise NotImplementedError('Normalized covariance unsupported')
            if pose_opt_plus is not None:
                pose_opt_plus = pnp_denormalize(transform, pose_opt_plus)
        return pose_opt, pose_cov, cost, pose_opt_plus

    def solve(self,
              x3d,
              x2d,
              w2d,
              camera,
              cost_fun,
              pose_init=None,
              cost_init=None,
              with_pose_cov=False,
              with_cost=False,
              force_init_solve=False,
              fast_mode=False):
        """
        Args:
            x3d (Tensor): Shape (num_obj, num_pts, 3)
            x2d (Tensor): Shape (num_obj, num_pts, 2)
            w2d (Tensor): Shape (num_obj, num_pts, 2)
            camera: Camera object of batch size (num_obj, )
            cost_fun: PnPCost object of batch size (num_obj, )
            pose_init (None | Tensor): Shape (num_obj, 4 or 7) in [x, y, z, yaw], optional
            cost_init (None | Tensor): Shape (num_obj, ), PnP cost of pose_init, optional
            with_pose_cov (bool): Whether to compute the covariance of pose_opt
            with_cost (bool): Whether to compute the cost of pose_opt
            force_init_solve (bool): Whether to force using the initialization solver when
                pose_init is not None
            fast_mode (bool): Fall back to Gauss-Newton for fast inference

        Returns:
            tuple:
                pose_opt (Tensor): Shape (num_obj, 4 or 7)
                pose_cov (Tensor | None): Shape (num_obj, 4, 4) or (num_obj, 6, 6), covariance
                    of local pose parameterization
                cost (Tensor | None): Shape (num_obj, )
        """
        with torch.no_grad():
            num_obj, num_pts, _ = x2d.size()
            tensor_kwargs = dict(dtype=x2d.dtype, device=x2d.device)

            if num_obj > 0:
                # if cost_fun is None, use evaluate_fun for cost_init
                # then for cost_min pose selection
                evaluate_fun = partial(evaluate_pnp,
                                       x3d=x3d,
                                       x2d=x2d,
                                       w2d=w2d,
                                       camera=camera,
                                       cost_fun=cost_fun,
                                       clip_jac=not fast_mode)

                if pose_init is None or force_init_solve:
                    assert self.init_solver is not None
                    if pose_init is None:
                        pose_init_solve, _, _ = self.init_solver.solve(
                            x3d,
                            x2d,
                            w2d,
                            camera,
                            cost_fun,
                            fast_mode=fast_mode)
                        pose_opt = pose_init_solve
                    else:
                        if cost_init is None:
                            cost_init = evaluate_fun(pose=pose_init,
                                                     out_cost=True)[1]
                        pose_init_solve, _, cost_init_solve = self.init_solver.solve(
                            x3d,
                            x2d,
                            w2d,
                            camera,
                            cost_fun,
                            with_cost=True,
                            fast_mode=fast_mode)
                        use_init = cost_init < cost_init_solve
                        pose_init_solve[use_init] = pose_init[use_init]
                        pose_opt = pose_init_solve
                else:
                    pose_opt = pose_init.clone()

                jac = torch.empty((num_obj, num_pts * 2, self.dof),
                                  **tensor_kwargs)
                residual = torch.empty((num_obj, num_pts * 2), **tensor_kwargs)
                cost = torch.empty((num_obj, ), **tensor_kwargs)

                if fast_mode:  # disable trust region
                    for i in range(self.num_iter):
                        evaluate_fun(pose=pose_opt,
                                     out_jacobian=jac,
                                     out_residual=residual,
                                     out_cost=cost)
                        jac_t = jac.transpose(
                            -1, -2)  # (num_obj, 4 or 6, num_pts * 2)
                        jtj = jac_t @ jac  # (num_obj, 4, 4) or (num_obj, 6, 6)
                        diagonal = torch.diagonal(jtj, dim1=-2,
                                                  dim2=-1)  # (num_obj, 4 or 6)
                        diagonal += self.eps  # add to jtj
                        gradient = jac_t @ residual.unsqueeze(-1)
                        if self.dof == 4:
                            pose_opt -= solve_wrapper(gradient,
                                                      jtj).squeeze(-1)
                        else:
                            step = -solve_wrapper(gradient, jtj).squeeze(-1)
                            pose_opt[..., :3] += step[..., :3]
                            pose_opt[..., 3:] = F.normalize(
                                pose_opt[..., 3:] +
                                (camera.get_quaternion_transfrom_mat(
                                    pose_opt[..., 3:])
                                 @ step[..., 3:, None]).squeeze(-1),
                                dim=-1)
                else:
                    evaluate_fun(pose=pose_opt,
                                 out_jacobian=jac,
                                 out_residual=residual,
                                 out_cost=cost)
                    jac_new = torch.empty_like(jac)
                    residual_new = torch.empty_like(residual)
                    cost_new = torch.empty_like(cost)
                    radius = x2d.new_full((num_obj, ),
                                          self.initial_trust_region_radius)
                    decrease_factor = x2d.new_full((num_obj, ), 2.0)
                    step_is_successful = x2d.new_zeros((num_obj, ),
                                                       dtype=torch.bool)
                    i = 0
                    while i < self.num_iter:
                        self._lm_iter(pose_opt, jac, residual, cost, jac_new,
                                      residual_new, cost_new,
                                      step_is_successful, radius,
                                      decrease_factor, evaluate_fun, camera)
                        i += 1
                    if with_pose_cov:
                        jac[step_is_successful] = jac_new[step_is_successful]
                        jtj = jac.transpose(-1, -2) @ jac
                        diagonal = torch.diagonal(jtj, dim1=-2,
                                                  dim2=-1)  # (num_obj, 4 or 6)
                        diagonal += self.eps  # add to jtj
                    if with_cost:
                        cost[step_is_successful] = cost_new[step_is_successful]

                if with_pose_cov:
                    pose_cov = torch.inverse(jtj)
                else:
                    pose_cov = None
                if not with_cost:
                    cost = None

            else:
                pose_opt = torch.empty((0, 4 if self.dof == 4 else 7),
                                       **tensor_kwargs)
                pose_cov = torch.empty(
                    (0, self.dof,
                     self.dof), **tensor_kwargs) if with_pose_cov else None
                cost = torch.empty(
                    (0, ), **tensor_kwargs) if with_cost else None

            return pose_opt, pose_cov, cost

    def _lm_iter(self, pose_opt, jac, residual, cost, jac_new, residual_new,
                 cost_new, step_is_successful, radius, decrease_factor,
                 evaluate_fun, camera):
        jac[step_is_successful] = jac_new[step_is_successful]
        residual[step_is_successful] = residual_new[step_is_successful]
        cost[step_is_successful] = cost_new[step_is_successful]

        # compute step
        residual_ = residual.unsqueeze(-1)
        jac_t = jac.transpose(-1, -2)  # (num_obj, 4 or 6, num_pts * 2)
        jtj = jac_t @ jac  # (num_obj, 4, 4) or (num_obj, 6, 6)

        jtj_lm = jtj.clone()
        diagonal = torch.diagonal(jtj_lm, dim1=-2,
                                  dim2=-1)  # (num_obj, 4 or 6)
        diagonal += diagonal.clamp(
            min=self.min_lm_diagonal, max=self.max_lm_diagonal
        ) / radius[:, None] + self.eps  # add to jtj_lm

        # (num_obj, 4 or 6, 1) = (num_obj, 4 or 6, num_pts * 2) @ (num_obj, num_pts * 2, 1)
        gradient = jac_t @ residual_
        step_ = -solve_wrapper(gradient, jtj_lm)

        # evaluate step quality
        pose_new = self.pose_add(pose_opt, step_.squeeze(-1), camera)
        evaluate_fun(pose=pose_new,
                     out_jacobian=jac_new,
                     out_residual=residual_new,
                     out_cost=cost_new)

        model_cost_change = -(step_.transpose(-1, -2) @ (
            (jtj @ step_) / 2 + gradient)).flatten()

        relative_decrease = (cost - cost_new) / model_cost_change
        torch.bitwise_and(relative_decrease >= self.min_relative_decrease,
                          model_cost_change > 0.0,
                          out=step_is_successful)

        # step accepted
        pose_opt[step_is_successful] = pose_new[step_is_successful]
        radius[step_is_successful] /= (
            1.0 -
            (2.0 * relative_decrease[step_is_successful] - 1.0)**3).clamp(
                min=1.0 / 3.0)
        radius.clamp_(max=self.max_trust_region_radius, min=self.eps)
        decrease_factor.masked_fill_(step_is_successful, 2.0)

        # step rejected
        radius[~step_is_successful] /= decrease_factor[~step_is_successful]
        decrease_factor[~step_is_successful] *= 2.0
        return

    def gn_step(self, x3d, x2d, w2d, pose, camera, cost_fun):
        residual, _, jac = evaluate_pnp(x3d,
                                        x2d,
                                        w2d,
                                        pose,
                                        camera,
                                        cost_fun,
                                        out_jacobian=True,
                                        out_residual=True)
        jac_t = jac.transpose(-1, -2)  # (num_obj, 4 or 6, num_pts * 2)
        jtj = jac_t @ jac  # (num_obj, 4, 4) or (num_obj, 6, 6)
        jtj = jtj + torch.eye(self.dof, device=jtj.device,
                              dtype=jtj.dtype) * self.eps
        # (num_obj, 4 or 6, 1) = (num_obj, 4 or 6, num_pts * 2) @ (num_obj, num_pts * 2, 1)
        gradient = jac_t @ residual.unsqueeze(-1)
        step = -solve_wrapper(gradient, jtj).squeeze(-1)
        return step

    def pose_add(self, pose_opt, step, camera):
        if self.dof == 4:
            pose_new = pose_opt + step
        else:
            pose_new = torch.cat(
                (pose_opt[..., :3] + step[..., :3],
                 F.normalize(
                     pose_opt[..., 3:] + (camera.get_quaternion_transfrom_mat(
                         pose_opt[..., 3:]) @ step[..., 3:, None]).squeeze(-1),
                     dim=-1)),
                dim=-1)
        return pose_new


class RSLMSolver(LMSolver):
    """
    Random Sample Levenberg-Marquardt solver, a generalization of RANSAC.
    Used for initialization in ambiguous problems.
    """

    def __init__(self, num_points=16, num_proposals=64, num_iter=3, **kwargs):
        super(RSLMSolver, self).__init__(num_iter=num_iter, **kwargs)
        self.num_points = num_points
        self.num_proposals = num_proposals

    def center_based_init(self, x2d, x3d, camera, eps=1e-6):
        x2dc = solve_wrapper(
            F.pad(x2d, [0, 1], mode='constant', value=1.).transpose(-1, -2),
            camera.cam_mats).transpose(-1, -2)
        x2dc = x2dc[..., :2] / x2dc[..., 2:].clamp(min=eps)
        x2dc_std, x2dc_mean = torch.std_mean(x2dc, dim=-2)
        x3d_std = torch.std(x3d, dim=-2)
        if self.dof == 4:
            t_vec = F.pad(x2dc_mean, [0, 1], mode='constant', value=1.) * (
                x3d_std[..., 1] /
                x2dc_std[..., 1].clamp(min=eps)).unsqueeze(-1)
        else:
            t_vec = F.pad(x2dc_mean, [0, 1], mode='constant', value=1.) * (
                math.sqrt(2 / 3) * x3d_std.norm(dim=-1) /
                x2dc_std.norm(dim=-1).clamp(min=eps)).unsqueeze(-1)
        return t_vec

    def solve(self, x3d, x2d, w2d, camera, cost_fun, **kwargs):
        with torch.no_grad():
            bs, pn, _ = x2d.size()

            if bs > 0:
                mean_weight = w2d.mean(dim=-1).reshape(1, bs, pn).expand(
                    self.num_proposals, -1, -1)
                inds = torch.multinomial(mean_weight.reshape(-1, pn),
                                         self.num_points).reshape(
                                             self.num_proposals, bs,
                                             self.num_points)
                bs_inds = torch.arange(bs, device=inds.device)
                inds += (bs_inds * pn)[:, None]

                x2d_samples = x2d.reshape(
                    -1, 2)[inds]  # (num_proposals, bs, num_points, 2)
                x3d_samples = x3d.reshape(
                    -1, 3)[inds]  # (num_proposals, bs, num_points, 3)
                w2d_samples = w2d.reshape(
                    -1, 2)[inds]  # (num_proposals, bs, num_points, 3)

                pose_init = x2d.new_empty(
                    (self.num_proposals, bs, 4 if self.dof == 4 else 7))
                pose_init[..., :3] = self.center_based_init(x2d, x3d, camera)
                if self.dof == 4:
                    pose_init[..., 3] = torch.rand(
                        (self.num_proposals, bs),
                        dtype=x2d.dtype,
                        device=x2d.device) * (2 * math.pi)
                else:
                    pose_init[..., 3:] = torch.randn(
                        (self.num_proposals, bs, 4),
                        dtype=x2d.dtype,
                        device=x2d.device)
                    q_norm = pose_init[..., 3:].norm(dim=-1)
                    pose_init[..., 3:] /= q_norm.unsqueeze(-1)
                    pose_init.view(-1, 7)[(q_norm < self.eps).flatten(),
                                          3:] = x2d.new_tensor([1, 0, 0, 0])

                camera_expand = camera.shallow_copy()
                camera_expand.repeat_(self.num_proposals)
                cost_fun_expand = cost_fun.shallow_copy()
                cost_fun_expand.repeat_(self.num_proposals)

                pose, _, _ = super(RSLMSolver, self).solve(
                    x3d_samples.reshape(self.num_proposals * bs,
                                        self.num_points, 3),
                    x2d_samples.reshape(self.num_proposals * bs,
                                        self.num_points, 2),
                    w2d_samples.reshape(self.num_proposals * bs,
                                        self.num_points, 2),
                    camera_expand,
                    cost_fun_expand,
                    pose_init=pose_init.reshape(self.num_proposals * bs,
                                                pose_init.size(-1)),
                    **kwargs)

                pose = pose.reshape(self.num_proposals, bs, pose.size(-1))

                cost = evaluate_pnp(x3d,
                                    x2d,
                                    w2d,
                                    pose,
                                    camera,
                                    cost_fun,
                                    out_cost=True)[1]

                min_cost, min_cost_ind = cost.min(dim=0)
                pose = pose[min_cost_ind, torch.arange(bs, device=pose.device)]

            else:
                pose = x2d.new_empty((0, 4 if self.dof == 4 else 7))
                min_cost = x2d.new_empty((0, ))

            return pose, None, min_cost


class AngularCentralGaussian(TorchDistribution):
    arg_constraints = {'scale_tril': constraints.lower_cholesky}
    has_rsample = True

    def __init__(self, scale_tril, validate_args=None, eps=1e-6):
        q = scale_tril.size(-1)
        assert q > 1
        assert scale_tril.shape[-2:] == (q, q)
        batch_shape = scale_tril.shape[:-2]
        event_shape = (q, )
        self.scale_tril = scale_tril.expand(batch_shape + (-1, -1))
        self._unbroadcasted_scale_tril = scale_tril
        self.q = q
        self.area = 2 * math.pi**(0.5 * q) / math.gamma(0.5 * q)
        self.eps = eps
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = value.expand(
            broadcast_shape(value.shape[:-1],
                            self._unbroadcasted_scale_tril.shape[:-2]) +
            self.event_shape)
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, value)
        half_log_det = self._unbroadcasted_scale_tril.diagonal(
            dim1=-2, dim2=-1).log().sum(-1)
        return M.log() * (-self.q / 2) - half_log_det - math.log(self.area)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        normal = _standard_normal(shape,
                                  dtype=self._unbroadcasted_scale_tril.dtype,
                                  device=self._unbroadcasted_scale_tril.device)
        gaussian_samples = _batch_mv(self._unbroadcasted_scale_tril, normal)
        gaussian_samples_norm = gaussian_samples.norm(dim=-1)
        samples = gaussian_samples / gaussian_samples_norm.unsqueeze(-1)
        samples[gaussian_samples_norm < self.eps] = samples.new_tensor(
            [1.] + [0. for _ in range(self.q - 1)])
        return samples


class EProPnPBase(torch.nn.Module, metaclass=ABCMeta):
    """
    End-to-End Probabilistic Perspective-n-Points.

    Args:
        mc_samples (int): Number of total Monte Carlo samples
        num_iter (int): Number of AMIS iterations
        normalize (bool)
        eps (float)
        solver (dict): PnP solver
    """

    def __init__(self,
                 mc_samples=512,
                 num_iter=4,
                 normalize=False,
                 eps=1e-5,
                 solver=None):
        super(EProPnPBase, self).__init__()
        assert num_iter > 0
        assert mc_samples % num_iter == 0
        self.mc_samples = mc_samples
        self.num_iter = num_iter
        self.iter_samples = self.mc_samples // self.num_iter
        self.eps = eps
        self.normalize = normalize
        self.solver = solver

    def forward(self, *args, **kwargs):
        return self.solver(*args, **kwargs)


class EProPnP6DoF(EProPnPBase):
    """
    End-to-End Probabilistic Perspective-n-Points for 6DoF pose estimation.
    The pose is parameterized as [x, y, z, w, i, j, k], where [w, i, j, k]
    is the unit quaternion.
    Adopted proposal distributions:
        position: multivariate t-distribution, degrees of freedom = 3
        orientation: angular central Gaussian distribution
    """

    def __init__(self, *args, acg_mle_iter=3, acg_dispersion=0.001, **kwargs):
        super(EProPnP6DoF, self).__init__(*args, **kwargs)
        self.acg_mle_iter = acg_mle_iter
        self.acg_dispersion = acg_dispersion
