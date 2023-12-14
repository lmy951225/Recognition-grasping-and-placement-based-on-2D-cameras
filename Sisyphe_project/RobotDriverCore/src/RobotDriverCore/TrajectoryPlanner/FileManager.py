from typing import Union, Dict
import os
import shutil
import sys
import json
import pandas

class FileManager():
    def __init__(self, save_dir: str) -> None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.__save_dir = save_dir

    @property
    def save_dir(self) -> str:
        self.__save_dir

    def clear_all_cache(self):
        if os.path.isdir(self.__save_dir):
            shutil.rmtree(self.__save_dir)
        os.makedirs(self.__save_dir)
    
    ##########################file operation##########################
    def __search_file(self, file_name: str) -> Union[str, bool]:
        file_path = os.path.join(self.__save_dir, file_name)
        return file_path if os.path.isfile(file_path) else False
    
    # json trajectory operation
    def save_json_trajectory(self, json_trajectory: Dict, trajectory_name: str) -> str:
        save_json_path = os.path.join(self.__save_dir, trajectory_name + ".json")
        with open(save_json_path, "w") as f:
            json.dump(json_trajectory, f)
        return save_json_path
    
    def delete_json_trajectory(self, trajectory_name: str) -> None:
        result = self.__search_file(trajectory_name + ".json")
        if result:
            os.remove(result)

    def search_json_trajectory(self, trajectory_name) -> Union[str, bool]:
        return self.__search_file(trajectory_name + ".json")
    
    # kinematics operation
    def save_trajectory_kinematics(self, trajectory_kinematics: Dict, trajectory_name: str) -> str:
        save_json_path = os.path.join(self.__save_dir, trajectory_name + "_kinematics.json")
        with open(save_json_path, "w") as f:
            json.dump(trajectory_kinematics, f)
        return save_json_path
    
    def delete_trajectory_kinematics(self, trajectory_name: str) -> None:
        result = self.__search_file(trajectory_name + "_kinematics.json")
        if result:
            os.remove(result)

    def search_trajectory_kinematics(self, trajectory_name) -> Union[str, bool]:
        return self.__search_file(trajectory_name + "_kinematics.json")
    
    # csv trajectory operation
    def save_csv_trajectory(self, csv_trajectory: pandas.DataFrame, trajectory_name: str) -> str:
        save_csv_path = os.path.join(self.__save_dir, trajectory_name + ".csv")
        csv_trajectory.to_csv(save_csv_path, index=False)
        return save_csv_path
    
    def delete_csv_trajectory(self, trajectory_name: str) -> None:
        result = self.__search_file(trajectory_name + ".csv")
        if result:
            os.remove(result)

    def search_csv_trajectory(self, trajectory_name) -> Union[str, bool]:
        return self.__search_file(trajectory_name + ".csv")
    
    # executable trajectory operation
    def delete_executable_trajectory(self, trajectory_name: str) -> None:
        result = self.__search_file(trajectory_name + ".trajectory")
        if result:
            os.remove(result)

    def search_executable_trajectory(self, trajectory_name: str) -> None:
        return self.__search_file(trajectory_name + ".trajectory")