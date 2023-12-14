import logging
from pathlib import Path
import numpy as np
import json

import triton_python_backend_utils as pb_utils
# from .pose_onnx_infer_triton import Pose_Estimation
from p10_infer import Pose_Estimation

# class PoseInfer(object):
#     def __init__(self) -> None:
#         self.box = np.loadtxt(Path(__file__).parent.joinpath("./test-box.txt"))
#         self.pose = np.loadtxt(Path(__file__).parent.joinpath("./test-pose2.txt"))
#         self.checkpoint_path = str(Path(__file__).parent.joinpath("./tmp.onnx"))
#         self.object_size_path = str(Path(__file__).parent.joinpath("./models_info.txt"))
#         self.camera_intrinsic_file = str(Path(__file__).parent.joinpath("./cameras.txt"))

#     def infer(self, input_img):
#         return pose_onnx_infer(
#             checkpoint_path = self.checkpoint_path,
#             input_img=input_img,
#             pose=self.pose,
#             box=self.box,
#             object_size_path=self.object_size_path,
#             camera_intrinsic_file=self.camera_intrinsic_file
#         )


class TritonPythonModel:

    def initialize(self, args):
        self.pose_onnx_infer = Pose_Estimation()
        self.model_config = model_config = json.loads(args['model_config'])

        self.out0_config = pb_utils.get_output_config_by_name(
            model_config, "world_rigid_body")
        self.out0_dtype = pb_utils.triton_string_to_numpy(
            self.out0_config['data_type'])

        self.out1_config = pb_utils.get_output_config_by_name(
            model_config, "pose_world")
        self.out1_dtype = pb_utils.triton_string_to_numpy(
            self.out1_config['data_type'])

        self.out2_config = pb_utils.get_output_config_by_name(
            model_config, "tcp_pose")
        self.out2_dtype = pb_utils.triton_string_to_numpy(
            self.out2_config['data_type'])

        # self.box = np.loadtxt(Path(__file__).parent.joinpath("./test-box.txt"))
        # self.pose = np.loadtxt(Path(__file__).parent.joinpath("./test-pose2.txt"))
        self.checkpoint_path = str(
            Path(__file__).parent.joinpath("./tmp.onnx"))
        # self.object_size_path = str(Path(__file__).parent.joinpath("./models_info.txt"))
        # self.camera_intrinsic_file = str(Path(__file__).parent.joinpath("./cameras.txt"))

        # self.net = PoseInfer()

    def execute(self, requests):
        responses = []

        for request in requests:
            logging.warn(
                "================request011==========================")
            in0 = pb_utils.get_input_tensor_by_name(request,
                                                    "input_img").as_numpy()
            in1 = pb_utils.get_input_tensor_by_name(request, "box").as_numpy()
            # print("================request1==========================")
            in2 = pb_utils.get_input_tensor_by_name(request, "pose").as_numpy()

            in3 = pb_utils.get_input_tensor_by_name(
                request, "flag").as_numpy()[0].decode("utf-8")
            # text = in_0[0].decode("utf-8")

            logging.warn(f"in0 {in0.shape}")
            logging.warn(f'in2 {in2}')
            # print("================request==========================")
            out0, out1, out2 = self.pose_onnx_infer.inference(
                img=in0,
                box=in1,
                pose_tcp_in_world=in2,
                flag=in3,
            )
            logging.warn("================request00==========================")
            # out0 = np.array([1, 2, 3], np.float32)
            out0_tensor = pb_utils.Tensor("world_rigid_body",
                                          out0.astype(self.out0_dtype))
            out1_tensor = pb_utils.Tensor("pose_world",
                                          out1.astype(self.out1_dtype))
            out2_tensor = pb_utils.Tensor("tcp_pose",
                                          out2.astype(self.out1_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out0_tensor, out1_tensor, out2_tensor])
            responses.append(inference_response)
            logging.warn("================request11==========================")
        return responses

    def finalize(self):
        print('Cleaning up...')
