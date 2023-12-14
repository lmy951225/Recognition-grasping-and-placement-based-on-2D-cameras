import argparse
import numpy as np
import sys
from codetiming import Timer
import cv2
import json
import tritonclient.grpc as grpcclient



def infer_client():
    try:
        triton_client = grpcclient.InferenceServerClient(
            url='localhost:8001',
            verbose=False,
            ssl=False,
            root_certificates=None,
            private_key=None,
            certificate_chain=None)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    # zidane.jpg bus
    model_name = "pose"



    statistics = triton_client.get_inference_statistics(model_name=model_name)
    # print(statistics)
    if len(statistics.model_stats) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)
    else:
        print(model_name, 'ok')

    # # Infer
    inputs = []
    outputs = []
    # bus zidane
    imgFile = "/workdir/models/pose/1/test-color.png"  # 读取文件的路径
    img_item = cv2.imread(imgFile, flags=1)  #np.ones((1082, 1937, 3), dtype="uint8")
    print(f'img_item {img_item.shape}')
    img= np.expand_dims(img_item,axis=0)#np.concatenate([img1, img1], axis=0)
    print(f'img {img.shape}')
    # input_data= np.concatenate([img, img], axis=0)
    input_data = img
    print(f'input_data {input_data.shape}')

    box_data=np.loadtxt('/workdir/models/pose/1/test-box.txt')
    box_data = np.expand_dims(box_data, axis=0)
    # box = np.expand_dims(box_data, axis=0)
    # print(box)
    print(f'box {box_data.shape}')

    pose_data = np.loadtxt('/workdir/models/pose/1/test-pose2.txt')
    pose_data = np.expand_dims(pose_data, axis=0)
    print(f'pose {pose_data.shape}')


    # box_data = np.ones((4, 4), dtype="float64")
    # pose_data = np.ones((input_data.shape[0],3, 4), dtype="float64")

    inputs.append(grpcclient.InferInput('input_img', input_data.shape, "UINT8"))
    inputs.append(grpcclient.InferInput('box', box_data.shape, "FP64"))
    inputs.append(grpcclient.InferInput('pose', pose_data.shape, "FP64"))

    print("input_data", input_data.shape)

    # # Initialize the data

    inputs[0].set_data_from_numpy(input_data)
    inputs[1].set_data_from_numpy(box_data)
    inputs[2].set_data_from_numpy(pose_data)
    # inputs[1].set_data_from_numpy()
    
    outputs.append(grpcclient.InferRequestedOutput('world_rigid_body'))
    outputs.append(grpcclient.InferRequestedOutput('pose_world'))
    outputs.append(grpcclient.InferRequestedOutput('tcp_pose'))
  
    # # Test with outputs
    with Timer("infer"):
        results = triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            client_timeout=None,
            headers={'test': '1'},
            compression_algorithm=None)

    output0_data = results.as_numpy('world_rigid_body')
    output1_data = results.as_numpy('pose_world')
    output2_data = results.as_numpy('tcp_pose')

    print(output0_data.shape)
    print(output1_data.shape)


if __name__ == '__main__':


    # print(output0_data.shape, output0_data.dtype)
    # print(output0_data[:, 5])
    # print(output0_data[:, 6])
    # # print(output1_data.shape, output1_data.dtype)
    # print(output1_data)

    # img_raw = cv2.imread(img_path)
    # # img_raw = input_data
    # colors = [
    #     (255, 0, 0), (0, 255, 0), (0, 0, 255),
    #     (255, 255, 0), (255, 0, 255), (0, 255, 255)
    # ]
    # for i, ele in enumerate(output0_data):
    #     label = str(output1_data[i])[1:]
    #     cv2.rectangle(img_raw, 
    #         (int(ele[1]), int(ele[2])), 
    #         (int(ele[3]), int(ele[4])), 
    #         colors[i % len(colors)], thickness=2)
    #     cv2.putText(img_raw, label, (int(ele[1]), int(ele[2])), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 
    #                1, (255,255,255), 1, cv2.LINE_AA)
    
    # cv2.imwrite("_out_.png", img_raw)
