import argparse
import numpy as np
import sys
# from codetiming import Timer
import cv2
import json
import tritonclient.grpc as grpcclient



def infer_client(img_path, box, transform, flag='0_0'):
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

    model_name = "pose"

    statistics = triton_client.get_inference_statistics(model_name=model_name)

    if len(statistics.model_stats) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)
    else:
        print(model_name, 'ok')

    # Infer
    inputs = []
    outputs = []
    img_item = cv2.imread(img_path, flags=1)
    if flag != "3_0_y_s":
        img_item = cv2.flip(img_item, -1)
    # print(f'img_item {img_item.shape}')
    input_data = np.expand_dims(img_item,axis=0)#np.concatenate([img1, img1], axis=0)

    box_data = np.expand_dims(box, axis=0)
    pose_data = np.expand_dims(transform, axis=0)

    inputs.append(grpcclient.InferInput('input_img', input_data.shape, "UINT8"))
    inputs.append(grpcclient.InferInput('box', box_data.shape, "FP64"))
    inputs.append(grpcclient.InferInput('pose', pose_data.shape, "FP64"))
    inputs.append(grpcclient.InferInput('flag',[1], "BYTES"))

    # Initialize the data

    inputs[0].set_data_from_numpy(input_data)
    inputs[1].set_data_from_numpy(box_data)
    inputs[2].set_data_from_numpy(pose_data)

    tmp = np.array([flag.encode("utf-8")],dtype=np.object_)
    inputs[3].set_data_from_numpy(tmp)
 
    outputs.append(grpcclient.InferRequestedOutput('world_rigid_body'))
    outputs.append(grpcclient.InferRequestedOutput('pose_world'))
    outputs.append(grpcclient.InferRequestedOutput('tcp_pose'))
  
    # Test with outputs
    results = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
        client_timeout=None,
        headers={'test': '1'},
        compression_algorithm=None)

    # output0_data = results.as_numpy('world_rigid_body')
    output1_data = results.as_numpy('pose_world')   # 物体位姿 / 目标检测结果为空
    output2_data = results.as_numpy('tcp_pose')     # TCP位姿态

    # print('pose_world----------------',output1_data)
    # print(output1_data.shape)

    return output2_data, output1_data


# if __name__ == '__main__':

# infer_client
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
