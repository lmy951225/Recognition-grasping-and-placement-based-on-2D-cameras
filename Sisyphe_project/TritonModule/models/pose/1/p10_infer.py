import argparse
import onnx
import onnxruntime as rt
import os
import os.path as osp
import cv2
import math
import numpy as np
from pathlib import Path
import logging
import torch
import copy

from torchvision.utils import save_image

from mmpose.apis import inference_topdown
from mmpose.structures import merge_data_samples

from p00_prep import xyz_rxryrz2transformation, get_parameter, undistort, xyxy_to_cs, zoom_in
from p01_yolo import yolo_detect
from p02_6dof import EProPnP6DoF, LMSolver, RSLMSolver
from p03_post import get_world_pose, Post_processing
from p04_calb import place_T_infer
from p05_visualize import show_pose, show_result

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
IMG = {"JPG", "jpg", "png", "bmp"}


class Pose_Estimation(object):

    def __init__(self, guide_type="inference", visualize=True):
        assert guide_type in ["inference", "cal"]
        self.file_path = os.path.dirname(os.path.abspath(__file__))
        self.visualize = visualize
        self.guide_type = guide_type
        self.start_model()
        self.get_parameter(
            para_path=str(Path(__file__).parent.joinpath("./parameter")))

    def start_model(self):
        """Load onnx model in memories by parameters dict.
        Define model structure before inference procedure.
        """
        checkpoint_path_pose_crawl = os.path.join(self.file_path, "tmp.onnx")
        onnx_model = onnx.load(checkpoint_path_pose_crawl)
        onnx.checker.check_model(onnx_model)
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))

        checkpoint_path_yolo = os.path.join(self.file_path, "yolo.onnx")
        onnx_model = onnx.load(checkpoint_path_yolo)
        onnx.checker.check_model(onnx_model)
        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input_yolo = list(set(input_all) - set(input_initializer))

        assert len(net_feed_input) == 1
        assert len(net_feed_input_yolo) == 1

        crawl_model = rt.InferenceSession(
            checkpoint_path_pose_crawl,
            providers=[
                # "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ] if rt.get_device() == "GPU" else [
                "CPUExecutionProvider",
            ])

        yolo_model = rt.InferenceSession(
            checkpoint_path_yolo,
            providers=[
                # "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ] if rt.get_device == "GPU" else [
                "CPUExecutionProvider",
            ])

        self.epropnp = EProPnP6DoF(mc_samples=512,
                                   num_iter=4,
                                   solver=LMSolver(dof=6,
                                                   num_iter=5,
                                                   init_solver=RSLMSolver(
                                                       dof=6,
                                                       num_points=16,
                                                       num_proposals=4,
                                                       num_iter=3)))
        # Ignore history version guide_model.
        # Because in inference procedure, guide model just a pose estimation model
        self.crawl_model = crawl_model
        self.yolo_model = yolo_model

    def get_parameter(self, para_path="./parameter"):
        """Get parameters, such as camera intrinsic, et.al

        :param para_path: parameter files path, default path is \"./parameter\"
        """
        self.para_dict = get_parameter(para_path)

    def concat_multi_box_results(self, img, box):
        """Concat multiple workpieces detection results in undistort img.
        If multiple objects, concat them for a batch.

        :param img: input image
        :param box: yolo boxes for zoom_in
        :returns
            one_img: zoom_in imgs org by axis 0
            box_center: box center location org as one_img
            box_scale: box scale org as as one_img
        """
        if len(box) == 1:
            return self.pose_image_preprocessing(img, box[0])
        processing_list = [
            self.pose_image_preprocessing(img, box[ind])
            for ind in range(len(box))
        ]
        img_list = [process_result[0] for process_result in processing_list]
        box_center_list = [
            process_result[1] for process_result in processing_list
        ]
        box_scale_list = [
            process_result[2] for process_result in processing_list
        ]
        logging.warning("box_center_list {}".format(box_center_list))
        one_img = torch.cat(img_list, 0)
        box_center = torch.cat(box_center_list, 0)
        box_scale = torch.cat(box_scale_list, 0)
        return one_img, box_center, box_scale

    def sort_boxes_along_axis(self, obj_imgs, box_centers, box_scales, axis):
        assert obj_imgs.shape[0] == box_centers.shape[0] == box_scales.shape[0]
        index = torch.tensor([idx for idx in range(box_centers.shape[0])])
        box_centers_with_index = torch.cat([box_centers, index.unsqueeze(1)], axis=1)
        # sort bbox along axis
        if axis == "x":
            axis_order = box_centers_with_index[:, 1].argsort()
        elif axis == "y":
            axis_order = box_centers_with_index[:, 0].argsort()
        else:
            axis_order = index
        obj_imgs = obj_imgs[axis_order]
        box_centers = box_centers[axis_order]
        box_scales = box_scales[axis_order]

        return obj_imgs, box_centers, box_scales

    def undistort(self, rgb_img):
        """Undistort img captured by robot-camera.

        :param rgb_img: input img
        :returns
            undistorted_img: undistorted image after intrinsic correction
        """
        # Undistort the full img with object
        camera_intrinsic = self.para_dict["camera_intrinsic"]
        distortion = self.para_dict["distortion"]
        width = self.para_dict["width"]
        height = self.para_dict["height"]
        undistorted_img = undistort(rgb_img, camera_intrinsic, distortion,
                                    width, height)

        cv2.imwrite("/models/undistorted_img2.png", undistorted_img)
        return undistorted_img

    def yolo(self, undistorted_img):
        """Yolo detection. Core det infer function in p01_yolo.

        :param distorted_img: input image for yolo detect
        :returns
            bboxes: detection bboxes.
        """
        pre = yolo_detect(undistorted_img,
                          self.yolo_model,
                          conf_threshold=0.5,
                          iou_threshold=0.45,
                          pre_treat_mode=2,
                          size_scale=3.24)
        logging.warn(f"pre {pre}")
        canvas = undistorted_img.copy()

        # yolo result visualization
        if len(pre.cpu().numpy()) == 0:
            logging.warn("No object has been detected")
            return
        else:
            box = pre.cpu().numpy()
            idx = []
            logging.warn(f"box {box.shape}")
            for i in range(box.shape[0]):
                bbox = box[i]
                tt = abs(int(bbox[2]) -
                         int(bbox[0])) / abs(int(bbox[3]) - int(bbox[1]))
                logging.warn(f"tt {tt}")
                if tt > 2 or tt < 0.5:
                    idx.append(i)
                    continue
                if self.visualize:
                    color = (255, 0, 0)
                    thickness = 3
                    canvas = cv2.rectangle(canvas,
                                           (int(bbox[0]), int(bbox[1])),
                                           (int(bbox[2]), int(bbox[3])),
                                           color,
                                           thickness=3,
                                           lineType=cv2.LINE_AA)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (int(bbox[0]), int(bbox[1]))
                    fontScale = 1

                    text = str(float(bbox[-1])) + "_" + str(float(bbox[-2]))
                    canvas = cv2.putText(canvas, text, org, font, fontScale,
                                         color, thickness, cv2.LINE_AA)
            logging.warn("self.visualize",self.visualize)
            if self.visualize:
                cv2.imwrite("/models/img_yolo_det.png", canvas)

            bboxes = np.delete(box, idx, axis=0)
            bboxes = bboxes[:, :4]

        return bboxes

    def pose_image_preprocessing(self, undistorted_img, box):
        """Input image preprocessing. Prepare img for camera 6dof estimation.

        :param undistorted_img: input image.
        :param box: object boxes detected by yoloV8.
        :returns
            one_img: zoom-in rgb images. Set object in the center.
        """
        pad_ratio = self.para_dict["pad_ratio"]
        height, width = undistorted_img.shape[:2]
        inp_res = self.para_dict["inp_res"]

        c, s = xyxy_to_cs(box, pad_ratio, s_max=max(width, height))
        rgb, c_h_, c_w_, s_ = zoom_in(undistorted_img, c, s, inp_res)
        rgb = rgb.transpose(2, 0, 1).astype(np.float32) / 255.
        one_img = torch.from_numpy(rgb).unsqueeze(0).float()
        box_center = torch.from_numpy(np.array([c_w_, c_h_]).reshape(1, 2))
        box_scale = torch.from_numpy(np.array(s_).reshape(1, ))
        box = torch.from_numpy(box).unsqueeze(0)

        return one_img, box_center, box_scale

    def pnp_inference(self,
                      img,
                      distorted_img,
                      box_center,
                      box_scale,
                      pose_tcp_in_world,
                      guide_model,
                      name="slect"):
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
            "object_size": self.para_dict["object_size"],
            "camera_intrinsic": self.para_dict["camera_intrinsic"],
            "epropnp": self.epropnp,
            "out_res": 64,
            "bs": img.shape[0]
        }
        output_names = [x.name for x in guide_model.get_outputs()
                        ]  # ['OUTPUT__0', 'OUTPUT__1']
        output_res = guide_model.run(
            output_names,
            {guide_model.get_inputs()[0].name: img.detach().numpy()})

        # epro-pnp start
        noc = output_res[0]
        w2d = output_res[1]
        # object poses in cam, pose^{camera}_{object}
        pose_est = Post_processing(torch.from_numpy(noc),
                                   torch.from_numpy(w2d), one_meta)
        # convert scale to cad
        pose_est[:, :3, -1] *= 1000
        logging.warn('-------++++++++++++++++++++++---------{}'.format(pose_est))

        
        # rgb_img = np.copy(distorted_img)

        
        pose_infer = np.vstack([pose_est[0], np.array([0, 0, 0, 1])])
        

        if pose_tcp_in_world.shape[0] * pose_tcp_in_world.shape[1] != 16:
            pose_tcp_in_world = np.vstack(
                [pose_tcp_in_world, [0, 0, 0, 1]])
            
        # pose_obj_in_world_ori = get_world_pose(pose_tcp_in_world,
        #                                     self.para_dict["cam_in_tcp"],
        #                                     pose_infer)
        # logging.warn("pose_obj_in_world {}".format(pose_obj_in_world_ori))

        # #adjust the pose of the object
        # T_cam_in_base = pose_tcp_in_world @ self.para_dict["cam_in_tcp"]
        # T_z = T_cam_in_base[:3,:3] @ pose_infer[:3,3]
        # logging.warn("T_z {}".format(T_z))
        # delt = (28 - T_z[2] - T_cam_in_base[2,3])/T_cam_in_base[2,2]
        # logging.warn("delt {}".format(delt))
        # pose_infer[2,3] += delt

        # pose_tcp means T^{world}_{tcp}
        # cam_in_tcp parameter means T_^{tcp}_{cam}
        # pose_infer means pose T^{cam}_{obj}, 6dof in camera coordinate system
        # finally got T^{world}_{obj}
        pose_obj_in_world = get_world_pose(pose_tcp_in_world,
                                            self.para_dict["cam_in_tcp"],
                                            pose_infer)
        
        
        # For debugging
        if self.visualize:
            position, position_w, _, center_t_w = show_pose(
                pose_infer[:3, :], self.para_dict["object_info"][1],
                (pose_tcp_in_world @ self.para_dict["cam_in_tcp"])[:3, :])
            show_result(distorted_img,
                        pose_infer,
                        position,
                        position_w,
                        center_t_w,
                        self.para_dict["camera_intrinsic"],
                        name=name)
        # poses_obj_in_world.append(pose_obj_in_world)

        return pose_obj_in_world
        # return pose_obj_in_world_ori,pose_obj_in_world

    def select_pnp(self, img, pose_tcp_in_world):
        """When first capture the object, do this func to guide grasp location.

        :param img: input image
        :param pose_tcp_in_world: origin tcp in world pose.
        :param guide_model: 6dof estimation model
        :return
            guided_poses_cam_in_world: multiple guided camera extrinsics to get the next pt initial location.
        """
        box = self.yolo(img)
        if box is None:
            logging.error("No workpieces in pose guidance procedure")
            raise ValueError("No boxes!")

        one_img, box_center, box_scale = self.concat_multi_box_results(img, box)
        one_img, box_center, box_scale = self.sort_boxes_along_axis(one_img, box_center, box_scale, axis=self.flag_camera_direction)

        pose_cam_in_world = pose_tcp_in_world @ self.para_dict["cam_in_tcp"]
        guided_pose_cam_in_world = copy.deepcopy(pose_cam_in_world)
        poses_obj_in_world, objs_center_in_world = None, None

        if self.guide_type == "inference":
            poses_obj_in_world = self.pnp_inference(one_img,
                                                    img,
                                                    box_center,
                                                    box_scale,
                                                    pose_tcp_in_world,
                                                    self.crawl_model,
                                                    name="crawl")
            logging.warn("poses_obj_in_world {}".format(poses_obj_in_world))
        elif self.guide_type == "cal":
            # convert obj in uv to obj in world
            # convert coord uv to [u, v, 1]
            center_in_uv = np.array(torch.cat([box_center, torch.ones(box_center.shape[0], 1)], axis=1).T)
            # Z_c decide by tcp initial height, grasp length and workpiece center height
            Z_c = 550 - 28 - self.para_dict["cam_in_tcp"][2, 3]
            objs_center_in_cam = Z_c * np.linalg.inv(self.para_dict["camera_intrinsic"]) @ center_in_uv
            logging.warn("objs_center_in_cam {}".format(objs_center_in_cam))
            # get multiple workpieces center in world
            objs_center_in_world = pose_tcp_in_world @ self.para_dict["cam_in_tcp"] @ np.vstack((objs_center_in_cam, np.ones(objs_center_in_cam.shape[1])))
            logging.warn("objs_center_in_world {}".format(objs_center_in_world))
        else:
            raise TypeError("Illegal guide type")

        guided_poses_cam_in_world = []
        nums = objs_center_in_world.shape[1] if self.guide_type == "cal" else len(poses_obj_in_world)

        for idx in range(nums):
            tx = objs_center_in_world[:, idx][0] if self.guide_type == "cal" else poses_obj_in_world[idx][0, 3]
            ty = objs_center_in_world[:, idx][1] if self.guide_type == "cal" else poses_obj_in_world[idx][1, 3]
            logging.warn("tx {} ty {}".format(tx, ty))
            if self.flag_camera_direction == "x":
                guided_pose_cam_in_world[0, 3] = tx
                guided_pose_cam_in_world[1, 3] = ty - math.cos(70 * math.pi / 180) * self.para_dict["select_r"]
                rz = 180
            elif self.flag_camera_direction == "y":
                guided_pose_cam_in_world[0, 3] = tx - math.cos(70 * math.pi / 180) * self.para_dict["select_r"]
                guided_pose_cam_in_world[1, 3] = ty
                rz = 90
                # logging.warn("guitx {} guity {}".format(guided_pose_cam_in_world[0, 3], guided_pose_cam_in_world[1, 3]))
            else:
                # capture the nearest scene
                dx = pose_cam_in_world[0, 3] - tx
                dy = pose_cam_in_world[1, 3] - ty
                r_xy = 310 * math.cos(70 / 180 * math.pi)
                r_d = math.sqrt(dx**2 + dy**2)

                ddx = dx * r_xy / (r_d + 0.00000001)
                ddy = dy * r_xy / (r_d + 0.00000001)

                rz = math.acos(ddy / r_xy) / math.pi * 180

                guided_pose_cam_in_world[0, 3] = tx + ddx
                guided_pose_cam_in_world[1, 3] = ty + ddy

            rx = 160
            ry = 0
            logging.warn("guitx {} ".format(guided_pose_cam_in_world))
            _guided_pose_cam_in_world = xyz_rxryrz2transformation(
                np.array([
                    guided_pose_cam_in_world[0, 3], guided_pose_cam_in_world[1, 3],
                    guided_pose_cam_in_world[2, 3], rx, ry, rz
                ])
            )
            guided_poses_cam_in_world.append(_guided_pose_cam_in_world)

        logging.warn("guided_pose_cam_in_world: {}".format(guided_poses_cam_in_world))
        return guided_poses_cam_in_world

    def crawl_pnp(self, img, pose_tcp_in_world, crawl_model):
        """Crawl pnp in pt2.

        :param img: input image.
        :param pose_tcp_in_world: initial pose tcp in the world.
        :param crawl_model: the pnp inference model. Default epropnp solver.
        :returns
            world_rigid_body: ignore.
            pose_obj_in_world: as declaration
            guided_pose_tcp_in_world: as above
        """
        box = self.yolo(img)
        if box is None:
            logging.error("No workpieces in pose guidance procedure")
            raise ValueError("No boxes!")

        one_img, box_center, box_scale = self.concat_multi_box_results(img, box)
        logging.warn("one_img {}".format(one_img.shape))
        # one_img, box_center, box_scale = self.sort_boxes_along_axis(one_img, box_center, box_scale, axis=self.flag_camera_direction)
        # one_img, box_center, box_scale = one_img[0].unsqueeze(0), box_center[0].unsqueeze(0), box_scale[0].unsqueeze(0)
        
        if self.visualize:
            save_image(one_img, "/models/one_img.png")

        index_center=torch.argsort(torch.abs(box_center[:,0]-1920//2))
        box_center=box_center[index_center][:1]
        
        one_img=one_img[index_center][:1]
        box_scale=box_scale[index_center][:1]

        # pose_obj_in_world_ori,pose_obj_in_world = self.pnp_inference(one_img,
        #                                         img,
        #                                         box_center,
        #                                         box_scale,
        #                                         pose_tcp_in_world,
        #                                         crawl_model,
        #                                         name="crawl")
        
        pose_obj_in_world = self.pnp_inference(one_img,
                                                img,
                                                box_center,
                                                box_scale,
                                                pose_tcp_in_world,
                                                crawl_model,
                                                name="crawl")
        logging.warn("pose_obj_in_world {}".format(pose_obj_in_world))
        
        
       

        # para_dict["paw_to_tcp"] means T^{paw}_{tcp}
        T_tcp_in_paw = np.linalg.inv(self.para_dict["paw_in_tcp"])
        # get guided_pose^{world}_{tcp} by T^{world}_{object} @ T^{object}_{grab_point} @ T^{grab_point}_{tcp}
        # self.para_dict["T_grasp_in_obj"] convert grab point to object, as obj in paw
        guided_pose_tcp_in_world = pose_obj_in_world @ self.para_dict[
            'T_grasp_in_obj'] @ T_tcp_in_paw
        logging.warn(
            'Guided pose tcp in world {}'.format(guided_pose_tcp_in_world))
        
        # guided_pose_tcp_in_world_ori = pose_obj_in_world_ori @ self.para_dict[
        #     'T_grasp_in_obj'] @ T_tcp_in_paw

        return np.ones([8, 3]), pose_obj_in_world, guided_pose_tcp_in_world
        # return np.ones([8, 3]), guided_pose_tcp_in_world, guided_pose_tcp_in_world_ori

    def select_pose(self, img, pose_tcp_in_world):
        """Select guided pose in pt1. Get a rough pose for grab in pt2.

        :param img: Input image.
        :param pose_tcp_in_world: Tcp extrinsics.
        :param guide_model: Default epropnp solver.
        :return
            guided_pose_tcp_in_world: guided extrinsics to get the next pt initial location.
        """
        # Move the camera to the estimated point.
        # Return the pose tcp in world.
        undistorted_img = self.undistort(img)
        pose_cam_in_world = self.select_pnp(undistorted_img, pose_tcp_in_world)
        T_tcp_in_cam = np.linalg.inv(self.para_dict["cam_in_tcp"])
        guided_pose_tcp_in_world = pose_cam_in_world @ T_tcp_in_cam

        return guided_pose_tcp_in_world

    def crawl_pose(self, img, pose_tcp_in_world, crawl_model):
        """Get 6d pose in pt2.

        :param img: input image.
        :param pose_tcp_in_world: initial tcp in world pose.
        :param crawl_model: pnp solver. Default epropnp LMSolver.
        :return
            world_rigid_body: Ignore.
            pose_obj_in_world: As declaration.
            guided_pose_tcp_in_world: As above. Then convert to tcp pose by grasp client.
        """
        # Return single tcp pose for paw procedure on pt2
        undistorted_img = self.undistort(img)
        world_rigid_body, pose_obj_in_world, guided_pose_tcp_in_world = self.crawl_pnp(
            undistorted_img, pose_tcp_in_world, crawl_model)
        logging.warn("pose_obj_in_world",pose_obj_in_world)
        return world_rigid_body, pose_obj_in_world, guided_pose_tcp_in_world

    def place_correction(self, img, pose):
        """place workpiece correction in pt3.

        :param img: Input image.
        :param pose: 6dof prediction
        :return
            T_correction: relation of transform to template
        """
        # Correct the pose during process 3
        mtx = np.array([[8532.78682392, 0., 1201.31253025],
                        [0., 8549.43490781, 1011.61631815], [0., 0., 1.]])

        dist = np.array(
            [[0.09077021, -5.99388352, -0.00113273, -0.00400096, 175.3684214]])

        undistort_img = cv2.undistort(img, mtx, dist)

        logging.warning("place correction: before pts2d inference")

        batch_results = inference_topdown(self.para_dict["pose_model"],
                                          undistort_img)
        logging.warning("place correction: after pts2d inference")
        results = merge_data_samples(batch_results)
        pts2d = results.pred_instances.keypoints
        logging.warning(f'pts2d----{pts2d}')
        logging.warning(f'pts2d shape----{pts2d.shape}')

        for idx_kp, point in enumerate(pts2d[0]):

            point[0] = max(min(point[0], 2048 - 1), 0)
            point[1] = max(min(point[1], 2448 - 1), 0)
            point = [int(point[0]), int(point[1])]
            undistort_img[point[0], point[1], :] = [0, 255, 0]
            undistort_img = cv2.circle(undistort_img, point, 10, (0, 0, 255),
                                       4)
            undistort_img = cv2.putText(undistort_img, str(idx_kp + 1), point,
                                        cv2.FONT_HERSHEY_SIMPLEX, 5,
                                        (255, 0, 0), 2)
        if self.visualize:
            cv2.imwrite("/models/circle.png", undistort_img)

        T_correction = place_T_infer(pts2d[0], tcp=pose)

        return T_correction

    def inference(self, img, box, pose_tcp_in_world, flag="0_0_y_s"):
        """Inference of all procedure. make up by flags.

        :param img: capture image.
        :param box: bbox of object.
        :param pose_tcp_in_world: initial tcp pose.
        :param flag: flags. Usage explaintion in code snippets.
        :return
            world_rigid_body: Ignore.
            pose_obj_in_world: as declaration.
            pose_tcp_in_world: as above.
        """
        # flag_camera_direction means grasp/tcp move along x/y direction, values "x" or "y"
        # pose_est_nums control the workpieces poses, set "s" will return just the nearest workpiece pose, default "s"
        self.flag_guide_pose, _, self.flag_camera_direction, _ = flag.split(
            "_")
        source_img = img[0]
        box = box[0]
        pose_tcp_in_world = pose_tcp_in_world[0]
        # logging.warn(f'pose============={pose}')
        self.para_dict["box"] = box
        self.para_dict["pad_ratio"] = 1.5
        # self.para_dict["pad_ratio"] = 1.3
        self.para_dict["inp_res"] = 256

        # Get world_rigid_body, pose_world and tcp_pose for different module
        world_rigid_body = np.array([[]], np.float32)
        poses_obj_in_world, guided_pose_tcp_in_world = None, None

        # For different flag, robot arm works different
        if self.flag_guide_pose == "1":
            # Get tcp pose for pre-grab
            # During this time, the robot arm turn to the place where the workpiece set
            try:
                poses_obj_in_world = np.array([[]], np.float32)
                # In new version, crawl_model is same as guide model
                # Get a initial rough pose, the the grasp move to a initial fixed height point
                guided_pose_tcp_in_world = self.select_pose(
                    source_img, pose_tcp_in_world
                )
                guided_pose_tcp_in_world[:,2, 3] = 449.754  #定高
            except Exception as e:
                logging.error("guide_pose error {}".format(e))
                return 1

        elif self.flag_guide_pose == "2":
            # Return final world && tcp 6dof results
            # Then turns to the client grasp opearition process
            try:
                _, poses_obj_in_world, guided_pose_tcp_in_world = self.crawl_pose(
                    source_img, pose_tcp_in_world, self.crawl_model
                )
            except Exception as e:
                logging.error("crawl_pose error {}".format(e))
                return 1

        elif self.flag_guide_pose == "3":
            # Correction in pt3
            try:
                poses_obj_in_world = np.array([[]], np.float32)
                guided_pose_tcp_in_world = self.place_correction(
                    source_img, pose_tcp_in_world)
            except Exception as e:
                logging.error("Place correction module error {}".format(e))
                return 1

        return world_rigid_body, poses_obj_in_world, guided_pose_tcp_in_world


def parse_args():
    parser = argparse.ArgumentParser(description="Convert MTL models to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/adt/data/1/objective-mtl/meta/shuiping_360.onnx",
        help="checkpoint file")
    parser.add_argument(
        "--checkpoint_yolo",
        type=str,
        default="/home/adt/data/1/6d_pose/models/pose/1/yolo.onnx",
        help="checkpoint file")
    parser.add_argument(
        "--input-img",
        type=str,
        default=
        "/home/adt/data/data/3d/6d_pose/byd116_0731/test/3_3_1_202307281531013578-color.png",
        help="Images for input")
    parser.add_argument("--cif",
                        type=str,
                        default="./cameras.txt",
                        help="camera intrinsic file")
    parser.add_argument("--osp",
                        type=str,
                        default="./models_info.txt",
                        help="object size path file")
    parser.add_argument("--opset-version", type=int, default=11)
    parser.add_argument(
        "--is-dynamic",
        action="store_true",
        help="whether using dynamic input and output",
    )
    parser.add_argument("--simplify",
                        action="store_true",
                        help="whether using onnx simplify")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the onnx model output against pytorch output",
    )
    parser.add_argument("--shape",
                        type=int,
                        nargs="+",
                        default=[416, 416],
                        help="input image size")
    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        default=[0, 0, 0],
        help="mean value used for preprocess input data",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        default=[255, 255, 255],
        help="variance value used for preprocess input data",
    )
    parser.add_argument(
        "--quantization",
        action="store_true",
        help="quantize model",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))
    try:
        args = parse_args()

        if not args.input_img:
            args.input_img = osp.join(osp.dirname(__file__),
                                      "../meta/test_data/a0519qvbyom_001.png")

        box = np.array([780.0, 203, float(780 + 426), float(203 + 426)])

        pose_tcp_in_world = np.loadtxt("./pose_tcp.txt")
        rgb_img = cv2.imread("./test_multi.jpg")
        rgb_img = cv2.flip(rgb_img, 1)
        cv2.imwrite("test.png", rgb_img)
        rgb_img = np.expand_dims(rgb_img, axis=0)
        pose_tcp_in_world = np.expand_dims(pose_tcp_in_world, axis=0)
        box = np.expand_dims(box, axis=0)

        pose_onnx_infer = Pose_Estimation(guide_type="cal", visualize=True)
        import time
        start = time.time()
        pose_onnx_infer.inference(rgb_img,
                                  box,
                                  pose_tcp_in_world,
                                  flag="1_0_y_s")
        end = time.time()
        logging.warn("time {}".format(end - start))

    except Exception as e:
        print("pth2onnx :", e)
        raise e
