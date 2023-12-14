
import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
import logging 
import os 
import sys
import time 
from utils.path_util import docker_host_path

def logging_config() -> None:
    
    docker_tmp_save_dir = os.path.join(docker_host_path, 'MotionPlanning')
    if not os.path.exists(docker_tmp_save_dir):
        os.makedirs(docker_tmp_save_dir)
    
    strftime = time.strftime("%Y-%m%d-%H:%M:%S", time.localtime())
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(
        filename = os.path.join(docker_tmp_save_dir, strftime + '.log'),
        level = logging.INFO,
        format = log_format
    )
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(consoleHandler)
    
loggingConfig = logging_config()
    
