import os 

abs_dir = os.path.dirname(os.path.abspath(__file__))

def robot_config_yaml_path(robot_type):

    yaml_path = os.path.join(abs_dir, "robot_config_yaml", robot_type + ".yaml")
 
    return yaml_path

def robot_urdf_path(robot_type):

    urdf_path = os.path.join(abs_dir, "robot_urdf", robot_type)
 
    return urdf_path