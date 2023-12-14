import numpy as np
import cv2
from utils_place.utils_rot import rotationMatrixToEulerAngles


def place_T_infer(pts2d, tcp=None):

    camera_Intrinsic = np.array([[8291.2396923, 0., 1307.06425739],
                                 [0., 8287.53309039, 1072.72472835],
                                 [0., 0., 1.]])

    points = [
        [592.126, -23.3586, 183.734],  # 1
        [571, -25.2696, 195.63],  # 2
        [549.874, -23.35785, 183.734],  # 3
        # a=np.array([548.792,-23.3578,183.734])
        # b=np.array([550.956,-23.3579,183.734])
        [587.096, 8.934095, 190.198],  # 4
        # a=np.array([588.253,8.93407,190.198])
        # b=np.array([585.939,8.93412,190.198])
        [554.904, 8.93467, 190.198],  # 5
        # a=np.array([556.061,8.93465,190.198])
        # b=np.array([553.747,8.93469,190.198])
        [571., 12.4607, 201.465],  # 6
        # a=np.array([571.372,12.4607,201.465])
        # b=np.array([570.628,12.4607,201.465])
        # c = (a + b) / 2
        # c
        [571.001, 20.5938, 201.465],  # 7
        [571.011, -0.231705, 225.802]  # 8
    ]
    #7个点
    # T_template=np.array([
    #     [  -0.01298953,   -0.99726035,   -0.07282218,   17.34847805],
    #    [   0.99989292,   -0.0124639 ,   -0.00766776, -569.41713757],
    #    [   0.00673911,   -0.07291398,    0.99731546,  109.45518524],
    #    [   0.        ,    0.        ,    0.        ,    1.        ]])
    #8个点
    T_template = np.array(
        [[-0.01282664, -0.99721014, -0.07353518, 17.39260928],
         [0.99988244, -0.01217353, -0.0093229, -569.09234575],
         [0.0084017, -0.07364611, 0.99724905, 108.53290369], [0., 0., 0., 1.]])
    model_points = np.array(points)
    success_2, rotation_vector_2, translation_vector_2 = cv2.solvePnP(
        model_points,
        pts2d,
        camera_Intrinsic,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not success_2:
        raise ValueError('PnP2')

    rotM_2 = cv2.Rodrigues(rotation_vector_2)[0]
    T_2 = np.concatenate([np.concatenate([rotM_2, translation_vector_2], axis=-1), \
                          np.array([[0, 0, 0, 1]])], axis=0)

    # T_cam2_in_cam1@T_1=T_2
    T_cam2_in_cam1 = T_2 @ np.linalg.inv(T_template)

    ########## compute from traj value

    T_tcp2_in_w_gt = tcp.copy()
    # rot_tcp2=eulerAngles2rotationMat(tcp[3:])
    # #################################
    # T_tcp2_in_w_gt=np.concatenate(\
    #     [np.concatenate([rot_tcp2,np.array(tcp[:3])[:,None]],axis=1),np.array([[0,0,0,1]])],axis=0)

    ###
    T_cam1_in_w = np.array(
        [[-0.01407914, 0.99989736, 0.00265485, 501.19911649],
         [-0.99930868, -0.01397939, -0.03444917, -316.44990866],
         [-0.03440852, -0.00313803, 0.99940292, 181.62336651],
         [0., 0., 0., 1.]])
    T_cam2_in_w = T_cam1_in_w @ T_cam2_in_cam1
    T_cam1_in_cam2 = np.linalg.inv(T_cam2_in_cam1)
    T_tcp1_in_w_predict = T_cam2_in_w @ T_cam1_in_cam2 @ np.linalg.inv(
        T_cam2_in_w) @ T_tcp2_in_w_gt

    print(T_tcp1_in_w_predict[:, 3])
    print(rotationMatrixToEulerAngles(T_tcp1_in_w_predict[:3, :3]))

    return T_tcp1_in_w_predict
