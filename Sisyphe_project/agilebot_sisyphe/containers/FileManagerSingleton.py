from typing import Union, Dict
import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
import json
import pandas
from Agilebot.IR.A.status_code import StatusCodeEnum
from Agilebot.IR.A.file_manager import FileManager
from utils.path_util import docker_tmp_trajectory_save_dir

class FileManagerSingleton():
    def __init__(self) -> None:
        self.__file_manager: Union[FileManager, None] = None
        self.__is_connected: bool = False

    def __new__(cls): # for singleton 
        if not hasattr(cls, 'instance'):
            cls.instance = super(FileManagerSingleton, cls).__new__(cls)
        return cls.instance

    @property
    def is_connected(self) -> bool:
        return self.__is_connected

    @property
    def file_manager(self) -> Union[FileManager, None]:
        return self.__file_manager
    
    def connect(self, controller_ip: str) -> StatusCodeEnum:
        self.__file_manager = FileManager(controller_ip=controller_ip)
        self.__is_connected = True
        return StatusCodeEnum.OK
    
    def disconnect(self):
        self.__is_connected = False
        self.__file_manager = None
    
    ###############################tracjectory operations#################################
    def __search_trajectory_file(self, trajectory_file_name: str) -> Union[str, bool]:
        file_path = os.path.join(docker_tmp_trajectory_save_dir, trajectory_file_name)
        return file_path if os.path.isfile(file_path) else False
            
    # local trajectory kinematics
    def upload_local_kinematics(self, kinematics: Dict, trajectory_name: str) -> str:
        save_kinematics_path = os.path.join(docker_tmp_trajectory_save_dir, trajectory_name + "_kinematics.json")
        with open(save_kinematics_path, 'w') as f:
            json.dump(kinematics, f)
        return save_kinematics_path
    
    def delete_local_kinematics(self, trajectory_name: str) -> str:
        result = self.__search_trajectory_file(trajectory_name + "_kinematics.json")
        if result:
            os.remove(result)
            return True
        else:
            return False
        
    def search_local_kinematics(self, trajectory_name: str) -> Union[str, bool]:
        return self.__search_trajectory_file(trajectory_name + "_kinematics.json")

    # local raw trajectory
    def upload_local_raw_trajectory(self, raw_trajectory: Dict, trajectory_name: str) -> str:
        save_raw_path = os.path.join(docker_tmp_trajectory_save_dir, trajectory_name + ".json")
        with open(save_raw_path, 'w') as f:
            json.dump(raw_trajectory, f)
        return save_raw_path

    def delete_local_raw_trajectory(self, trajectory_name: str) -> bool:
        result = self.__search_trajectory_file(trajectory_name + ".json")
        if result:
            os.remove(result)
            return True
        else:
            return False
    
    def search_local_raw_trajectory(self, trajectory_name: str) -> Union[str, bool]:
        return self.__search_trajectory_file(trajectory_name + ".json")
    
    # local detail trajectory
    def upload_local_detail_trajectory(self, detail_trajectory: pandas.DataFrame, trajectory_name: str):
        save_detail_path = os.path.join(docker_tmp_trajectory_save_dir, trajectory_name + ".csv")
        detail_trajectory.to_csv(save_detail_path, index=False)
        return save_detail_path
    
    def delete_local_detail_trajectory(self, trajectory_name: str) -> bool:
        result = self.__search_trajectory_file(trajectory_name + ".csv")
        if result:
            os.remove(result)
            return True
        else:
            return False
    
    def search_local_detail_trajectory(self, trajectory_name: str) -> Union[str, bool]:
        return self.__search_trajectory_file(trajectory_name + ".csv")

# only import this instance
FileManagerInstance = FileManagerSingleton()

if __name__ == "__main__":
    a = {1:2, "3":"4"}
    print(FileManagerInstance.upload_local_raw_trajectory(a, "test0"))
    print(FileManagerInstance.search_local_raw_trajectory("test0"))
    print(FileManagerInstance.delete_local_raw_trajectory("test0"))
    b = pandas.DataFrame([[1,2,3], ["a","b","c"]])
    print(FileManagerInstance.upload_local_detail_trajectory(b, "test1"))
    print(FileManagerInstance.search_local_detail_trajectory("test1"))
    print(FileManagerInstance.delete_local_detail_trajectory("test1"))
