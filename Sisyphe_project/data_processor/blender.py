import os.path
import bpy
import numpy as np
import cv2

def rotation_matrix_to_quaternion(R):
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    q0 = np.sqrt(1 + r11 + r22 + r33) / 2
    q1 = (r32 - r23) / (4 * q0)
    q2 = (r13 - r31) / (4 * q0)
    q3 = (r21 - r12) / (4 * q0)

    return q0, q1, q2, q3


def quaternion_to_xyz(quaternion):
    q0, q1, q2, q3 = quaternion

    r11 = 2 * q0 * q0 + 2 * q1 * q1 - 1
    r12 = 2 * q1 * q2 - 2 * q0 * q3
    r13 = 2 * q1 * q3 + 2 * q0 * q2
    r21 = 2 * q1 * q2 + 2 * q0 * q3
    r22 = 2 * q0 * q0 + 2 * q2 * q2 - 1
    r23 = 2 * q2 * q3 - 2 * q0 * q1
    r31 = 2 * q1 * q3 - 2 * q0 * q2
    r32 = 2 * q2 * q3 + 2 * q0 * q1
    r33 = 2 * q0 * q0 + 2 * q3 * q3 - 1

    rx = np.arctan2(r32, r33)
    ry = np.arctan2(-r31, np.sqrt(r32 * r32 + r33 * r33))
    rz = np.arctan2(r21, r11)

    return rx, ry, rz


def quaternion_to_rotation_matrix(quaternion):
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion

    rotation_matrix = np.array([[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                                [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
                                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]])

    return rotation_matrix

def get_filenames_by_character(directory, character):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if character in file:
                return directory, file
#                return os.path.join(directory, file)


def campara_to_camdata(path):
    f = open(path, "r")
    data = f.readlines()
    data = data[0:]
    return data


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return str(path)


def render_in_blender(data, directory):
    markers = {}

    # delete cameras
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete()

    for i, one_cam in enumerate(data[0:]):

        if i % 1 == 0:
            # set a camera
            cam_data = bpy.data.cameras.new(name='CamData')

            # cam intrinsic parameter
#            cam_data.lens = 50
#            # cam_data.sensor_fit = 'HORIZONTAL'
#            cam_data.sensor_fit = 'AUTO'
#            cam_data.sensor_width = 61
            cam_data = bpy.data.cameras.new(name='CamData')
            matrix = np.array([[1693.73105905,    0.,          996.42376472],
                               [   0.,         1693.31626151,  565.55906603],
                               [   0.,            0.,            1.        ]])
#            cam_data.camera_matrix = matrix

            # cam intrinsic parameter
            
            sensor_width = 60  # 传感器宽度，单位：毫米
#            sensor_height = 24  # 传感器高度，单位：毫米
            image_width = 1920  # 图像宽度，单位：像素
            image_height = 1080  # 图像高度，单位：像素
            cam_data.lens = (matrix[0, 0] + matrix[1, 1]) / 2 * (sensor_width/image_width)
            cam_data.lens = (matrix[0, 0] + matrix[1, 1]) / 2 * (sensor_width/image_width)
            cam_data.lens = (matrix[0, 0] + matrix[1, 1]) / 2 * (sensor_width/image_width)
            # 设置相机的传感器尺寸
            cam_data.sensor_width = sensor_width
            cam_data.shift_x = (image_width/2 - matrix[0,2]) / matrix[0, 0]
            cam_data.shift_y = (matrix[1, 2] - image_height/2) / matrix[1, 1]

            one_cam = one_cam.split(' ')
            cam_num = int(one_cam[-1].split('_')[1])
            cam = bpy.data.objects.new('Cam_{}'.format(cam_num), cam_data)
            bpy.context.scene.collection.objects.link(cam)

            cam_loc = np.array([one_cam[4:-1]]).astype(float)
            cam_quat = np.array(one_cam[0:4]).astype(float)

            r = quaternion_to_rotation_matrix(cam_quat)
            r_t = r.transpose()

            t = -1 * np.matmul(r_t, cam_loc.transpose())
            cam.location = np.array(t)#/1000

            cam_quat_2 = rotation_matrix_to_quaternion(r_t)
            cam.rotation_mode = 'QUATERNION'
            cam.rotation_quaternion = cam_quat_2

            cam.scale = [1/20, -1/20, -1/20]
            
            # 指定目录和字符
            character = '_' + str(cam_num) + '_1'
            directory, filepath = get_filenames_by_character(directory, character)
            img = bpy.data.images.load(os.path.join(directory, filepath))
            cam_data.show_background_images = True
            cam_data.show_background_images = True
            bg = cam_data.background_images.new()
            bg.image = img

            context = bpy.context
            scene = context.scene

            scene.camera = scene.objects.get('Cam_{}'.format(cam_num))

            # render
            bpy.context.scene.render.resolution_percentage = 100  # make sure scene height and width are ok (edit)
            bpy.ops.render.render()

            pixels = np.array(bpy.data.images['Viewer Node'].pixels)
            width = bpy.context.scene.render.resolution_x
            height = bpy.context.scene.render.resolution_y
            depth = np.reshape(pixels, (height, width, 4))
            depth = cv2.flip(depth, 0)
            depth = depth[:, :, 0]

            retval, image = cv2.threshold(depth, 250, 255, cv2.THRESH_BINARY)
            mask = (255 - image) / 255
            depth = cv2.bitwise_and(depth, depth, mask=mask.astype('uint8'))
            np.save(os.path.join(depth_path, '{}.npy'.format(one_cam[-1].split('.')[0])), depth)
            cv2.imwrite(os.path.join(mask_path, '{}.png'.format(one_cam[-1].split('.')[0])), mask)


if __name__ == "__main__":
    cam_extpara_path = "/media/jingjing/56a1e372-b7ba-43a2-8adc-ce445bc31fff/jingjing/share/data/bydGT_1107/dataset_270/images_trans.txt"
    projetc_dir = '/media/jingjing/56a1e372-b7ba-43a2-8adc-ce445bc31fff/jingjing/share/data/bydGT_1107/dataset_270/blender/'
    directory = '/media/jingjing/56a1e372-b7ba-43a2-8adc-ce445bc31fff/jingjing/share/data/bydGT_1107/dataset_270/images/'  # 替换为你的目录路径
    
    mask_path = make_dir(os.path.join(projetc_dir, 'masks'))
    depth_path = make_dir(os.path.join(projetc_dir, 'depths'))

    data = campara_to_camdata(cam_extpara_path)
    render_in_blender(data, directory)
