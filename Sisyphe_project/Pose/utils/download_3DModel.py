import boto3
import getpass
import os 
import logging
import zipfile

def download_file(src, dst, endpoint_url, username, password):
    
    s3 = boto3.resource('s3', 
                        endpoint_url=endpoint_url,
                        aws_access_key_id = username,
                        aws_secret_access_key = password)
    s3_bucket = s3.Bucket("lupinus")
    s3_bucket.download_file(src, dst)
    
def download_3DModel(parameters, _3DModel_path):
    
    EndEffectorType = parameters["EndEffectorType"]
    ManipulatorType = parameters["ManipulatorType"]
    StaticObstacleTypeList = parameters["StaticObstacleTypeList"]
    DynamicObstacleTypeList = parameters["DynamicObstacleTypeList"]
    endpoint_url = "http://fat-minio.idmaic.com:8090/"
    end_effector_path = os.path.join(_3DModel_path, "EndEffector", EndEffectorType + ".stl")
    robot_urdf_path = os.path.join(_3DModel_path, "RobotURDF", ManipulatorType + ".zip")
    static_Obstacle_path_list = [os.path.join(_3DModel_path, "StaticObstacle", type + ".stl") for type in  StaticObstacleTypeList]
    dynamic_Obstacle_path_list = [os.path.join(_3DModel_path, "DynamicObstacle", type + ".stl") for type in  DynamicObstacleTypeList]
    _3DM_path_list = static_Obstacle_path_list + dynamic_Obstacle_path_list + [end_effector_path] + [robot_urdf_path]
    for folder in ["EndEffector", "StaticObstacle", "DynamicObstacle", "RobotURDF"]:
        if not os.path.exists(os.path.join(_3DModel_path, folder)):
            os.makedirs(os.path.join(_3DModel_path, folder))
    is_exists_list = []
    for _3DM_path in _3DM_path_list:
        if not os.path.exists(_3DM_path):
            is_exists_list.append(False)
        else:
            is_exists_list.append(True)
    if sum(is_exists_list) != len(is_exists_list):
        logging.info("Some 3DModels do not exist and need to be downloaded now.")
        logging.info("Please enter the account password below to download the model file")
        # username = input("Enter your MinIO username: ")
        # password = getpass.getpass("Enter your password: ")
        username = "lupinus"
        password = "Lps#Wy123"
        for is_exists, _3DM_path in zip(is_exists_list, _3DM_path_list):
            if not is_exists:
                _3DM_name = os.path.basename(_3DM_path)
                src = os.path.join("flyshot_simulator/3DModel", _3DM_name)
                try:
                    download_file(src=src, dst=_3DM_path, endpoint_url=endpoint_url, username=username, password=password)
                except:
                    raise ValueError("Unable to find the file named {} in the MinIO. "
                                    "Please make sure that the {} file exists in MinIO and "
                                    "check that the type is spelled correctly".format(_3DM_name, _3DM_name))
                if _3DM_name.endswith(".zip"):
                    f = zipfile.ZipFile(_3DM_path, "r")
                    for file in f.namelist():
                        os.path.dirname
                        f.extract(file, os.path.dirname(_3DM_path))
                    f.close()       
        logging.info("The 3DModel download is finished.")  
    else:
        logging.info("The required 3DModel already exists, don't need to download")
            

