import numpy as np
import cv2
import os
import re


def read_camera_instrinsic(camera_intrinsic_file):
    with open(camera_intrinsic_file) as fr:
        camera_infos = fr.readlines()[3].split()
        fx = float(camera_infos[4])
        fy = float(camera_infos[5])
        cx = float(camera_infos[6])
        cy = float(camera_infos[7])
        camera_intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1.0]
        ], dtype=np.float32)
        d = np.array([float(camera_infos[idx]) for idx in range(8,13)])
    return camera_intrinsic, d

if __name__ == "__main__":
    camera_intrinsic_file = '/media/jingjing/56a1e372-b7ba-43a2-8adc-ce445bc31fff/jingjing/share/data/bydGT_1101/cameras.txt'
    image_path_circle = '/media/jingjing/56a1e372-b7ba-43a2-8adc-ce445bc31fff/jingjing/share/data/bydGT_1107/all/byd_circle_315/'
    image_path_sphere = '/media/jingjing/56a1e372-b7ba-43a2-8adc-ce445bc31fff/jingjing/share/data/bydGT_1107/all/byd_sphere_315/'
    save_path = '/media/jingjing/56a1e372-b7ba-43a2-8adc-ce445bc31fff/jingjing/share/data/bydGT_1107/dataset_315/images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    camera_intrinsic, d = read_camera_instrinsic(camera_intrinsic_file)
    files = os.listdir(image_path_circle)
    for file in files:
        img = cv2.imread(image_path_circle + file)
        img_m = cv2.flip(img, -1)
        undistorted_image = cv2.undistort(img_m, camera_intrinsic, d)
        cv2.imwrite(save_path +file, undistorted_image)

    
    files = os.listdir(image_path_sphere)
    for file in files:
        img = cv2.imread(image_path_sphere + file)
        img_m = cv2.flip(img, -1)
        undistorted_image = cv2.undistort(img_m, camera_intrinsic, d)
        num = int(file.split('_')[1]) + 360
        # print(num)
        nm = re.sub(r'1_\d*_1_', '1_'+str(num)+'_1_', file)
        cv2.imwrite(save_path+nm, undistorted_image)