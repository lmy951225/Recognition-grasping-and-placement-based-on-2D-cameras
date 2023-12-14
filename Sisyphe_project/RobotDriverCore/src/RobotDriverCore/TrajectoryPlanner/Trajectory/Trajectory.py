from typing import Dict
import json

class Trajectory():
    def __init__(self, trajectory_name: str, trajectory_path: str) -> None:
        self.__trajectory_name = trajectory_name 
        self.__trajectory_path = trajectory_path
        with open(trajectory_path, "r") as f:
            self._trajectory_content = json.load(f)

    @property
    def trajectory_name(self) -> str:
        return self.__trajectory_name
    
    @property
    def trajectory_path(self) -> str:
        return self.__trajectory_path
    
    @property
    def trajectory_content(self) -> Dict:
        return self._trajectory_content