from typing import List
from setuptools import setup
from Cython.Build import cythonize
import os
import shutil

root_dir: str = os.path.dirname(os.path.abspath(__file__))
file_paths: List[str] = list()

def collect_py_files(current_dir: str, path_containers: List[str]) -> List[str]:
    for file_name in os.listdir(current_dir):
        current_path = os.path.join(current_dir, file_name)
        if os.path.isfile(current_path) and current_path.endswith('py'):
            path_containers.append(current_path)
        elif os.path.isdir(current_path):
            collect_py_files(current_path, path_containers)

collect_py_files(current_dir=os.path.join(root_dir, 'RobotDriverCore'), path_containers=file_paths)
print(file_paths)
setup(ext_modules=cythonize(file_paths, language_level="3"))

for path in file_paths:
    os.remove(path.replace(".py", ".c"))

build_dir = os.path.join(root_dir, "build")
build_so_dir = os.path.join(build_dir, "lib.linux-x86_64-cpython-310/RobotDriverCore")

save_robot_driver_core_dir = os.path.join(build_dir, "RobotDriverCore")
shutil.copytree(src=build_so_dir, dst=save_robot_driver_core_dir)

# py2so procedure for docker deployment
# 1. cd /RobotDriverCore/src
# 2. docker run -it --rm -v $PWD:/home/adt hub.micro-i.com.cn:9443/eastwind/universalrobotcontrol:v0.0.*
# 3. python3 setup_docker_deployment.py build_ext 
