from typing import Union
import os
import time
import sys
import logging

utils_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(utils_dir)
# load parameters for docker deplolyment and local debugging
docker_host_path = os.path.join(os.path.expanduser('~'), "host_dir")
# for docker use only
# ceate tmp save dir
docker_tmp_save_dir = os.path.join(docker_host_path, 'robot_controller_tmp')
if not os.path.exists(docker_tmp_save_dir):
    os.makedirs(docker_tmp_save_dir)
# create log dir 
docker_tmp_log_save_dir = os.path.join(docker_tmp_save_dir, 'log')
if not os.path.exists(docker_tmp_log_save_dir):
    os.makedirs(docker_tmp_log_save_dir)
########################## set logging format ##################################
log_format = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(filename=os.path.join(docker_tmp_log_save_dir, time.strftime("%Y-%m%d-%H:%M:%S", time.localtime()) + '.log'), 
                    level=logging.WARN, 
                    format=log_format)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(consoleHandler)
logging.warning(
"""
            ||      || 
            ||      ||
            ||======||           
            ||      || 
            ||      ||
            ||      || 
            ||======||           
            ||      || 
            ||      ||
            ||      || 
            ||======||           
            ||      ||
    Powered by Ladder Project Lab.
    RobotController Version {}.
          """.format("0.0.8"))


###############################################
# create trajectory directory
docker_tmp_trajectory_save_dir = os.path.join(docker_tmp_save_dir, 'trajectory')
if not os.path.exists(docker_tmp_trajectory_save_dir):
    os.makedirs(docker_tmp_trajectory_save_dir)
