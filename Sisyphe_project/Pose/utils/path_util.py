"""
@Author      :   XiaoZhiheng
@Time        :   2023/01/31 10:33:44
"""

import os 
from .paramaters_utils import MachineTranslationType

utils_dir = os.path.dirname(os.path.abspath(__file__))
docker_host_path = os.path.join(os.path.expanduser('~'), "host_dir")

def get_parameters_path() -> str:
    
    parameters_local_path = os.path.join(utils_dir, "config", MachineTranslationType, 'parameters.json')
    parameters_docker_path = os.path.join(docker_host_path, "docker_config", "parameters.json")
    parameters_path = parameters_docker_path if os.path.exists(parameters_docker_path) else parameters_local_path
    
    return parameters_path    

def get_3DModel_path() -> str:
    
    model_path = os.path.join(docker_host_path, "docker_config", "3DModel")
    
    return model_path 







