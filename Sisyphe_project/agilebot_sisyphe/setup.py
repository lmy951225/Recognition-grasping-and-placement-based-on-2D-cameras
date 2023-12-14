from typing import List
from setuptools import setup
from Cython.Build import cythonize
import os
import shutil

root_dir: str = os.path.dirname(os.path.abspath(__file__))
file_paths: List[str] = list()

def collect_py_files(current_dir: str) -> List[str]:
    paths: List[str] = list()
    for file_name in os.listdir(current_dir):
        if file_name.endswith(".py"):
            paths.append(os.path.join(current_dir, file_name))
    return paths


# setup files
file_paths.extend(collect_py_files(current_dir=os.path.join(root_dir, "containers")))
file_paths.extend(collect_py_files(current_dir=os.path.join(root_dir, "trajectories")))
file_paths.extend(collect_py_files(current_dir=os.path.join(root_dir, "utils")))
print(file_paths)
setup(ext_modules=cythonize(file_paths, language_level="3"))

for path in file_paths:
    path1 = path.replace(".py", ".c")
    os.remove(path1)

build_dir = os.path.join(root_dir, "build")
build_so_dir = os.path.join(build_dir, "lib.linux-x86_64-cpython-38")

save_grpc_dir = os.path.join(build_so_dir, "grpc_module")
shutil.copytree(src=os.path.join(root_dir, "grpc_module"), dst=save_grpc_dir)

save_parameters_dir = os.path.join(build_so_dir, "DockerDeployment")
shutil.copytree(src=os.path.join(root_dir, "DockerDeployment"), dst=save_parameters_dir)

save_agilebot_controller_dir = os.path.join(build_dir, "AgilebotController")
shutil.copytree(src=build_so_dir, dst=save_agilebot_controller_dir)

# py2so procedure
# docker run -it --rm -v $PWD:/home/adt hub.micro-i.com.cn:9443/eastwind/agilebot:v0.08
# python3 setup.py build_ext 