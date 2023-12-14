import os
import numpy as np
import torch
import onnxruntime as rt
import cv2

from p00_prep import load_lm_model_info


def show_bbox(img, pred, save_path):
    # 目标检测结果可视化
    box = pred.cpu().numpy()
    bbox = box[0]
    color = (255, 0, 0)
    thickness = 3
    img_res = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            color,
                            thickness=3,
                            lineType=cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (int(bbox[0]), int(bbox[1]))
    fontScale = 1

    text = str(float(bbox[-1])) + "_" + str(float(bbox[-2]))
    img_res = cv2.putText(img_res, text, org, font, fontScale, color,
                          thickness, cv2.LINE_AA)
    cv2.imwrite(save_path, img_res)


def set_model_onnx(onnx_path):
    checkpoint_path = os.path.join(onnx_path)
    model = rt.InferenceSession(checkpoint_path,
                                providers=[
                                    "CPUExecutionProvider",
                                ])

    return model


def prepare_img_4_6dof(img):

    img = np.swapaxes(img, 0, -1)
    img = np.swapaxes(img, 1, -1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).cuda()
    img = img[None, :, :, :]

    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    return img


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
