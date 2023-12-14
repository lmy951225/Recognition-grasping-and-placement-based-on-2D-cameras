
from Pose.utils.tools import read_json
from Pose.utils.path_util import get_parameters_path
import numpy as np 

HomeJoints = np.array(read_json(get_parameters_path())["HomeJoints"])