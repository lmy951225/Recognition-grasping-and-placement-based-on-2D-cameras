import logging
import os
import sys
sys.path.append("../src")
import time
log_format = '%(asctime)s %(levelname)s %(message)s'
log_dir = os.path.join(os.getcwd(), "logger")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(filename=os.path.join(log_dir, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log'), 
                    level=logging.INFO, 
                    format=log_format)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(consoleHandler)

from RobotDriverCore.ManipulatorDriver.JAKADriver.JAKA_Zu7 import JAKA_Zu7
from RobotDriverCore.RobotDriverCoreUtils.RobotPosition import JointPosition, CartesianPosition
from gripper3 import gripper_client

def move_line_bias(bias_x,bias_y,bias_z,jaka_move_parameters):
    current_cartesian = jaka_zu7.get_current_cartesian_position()
    target_cartesian = CartesianPosition(xyzrxryrz=[current_cartesian.x + bias_x,current_cartesian.y + bias_y,current_cartesian.z+bias_z,
                                                    current_cartesian.rx,current_cartesian.ry,current_cartesian.rz])
    move_cartesian_response =jaka_zu7.move_line_sync(target_cartesian,jaka_move_parameters)
    if move_cartesian_response.code != 0:
        print("move_line faild")
        return False
    else:
        print("move_line successed")
        return True
    
    
if __name__ == "__main__":
    #夹爪
    lmy = gripper_client(port="/dev/ttyUSB1")
    #实例化对象
    jaka_zu7 = JAKA_Zu7()
    #连接
    assert jaka_zu7.connect(controller_ip="192.168.13.133")
    #上电
    jaka_zu7.power_on()
    #设置速度
    jaka_zu7.set_speed_ratio(speed_ratio=0.2)
    #上使能
    jaka_zu7.enable() 
    jaka_move_parameters_joint = {
        "total_wait_time":100,
        "speed":50,
        "accel":50,
    }
    jaka_move_parameters_cartesian = {
        "total_wait_time":100,
        "speed":100,
        "accel":200,
        "tol":0.2,
    }
    home_joint_position = JointPosition(joints=[91.144379, 92.363768, -17.412272, 104.48391, 91.960157, -90.006056]) #预设点
    move_joint_response = jaka_zu7.move2joint_position_sync(joint_position=home_joint_position, jaka_move_parameters=jaka_move_parameters_joint)
    if move_joint_response.code == 0:
        print("home point stand by")
    target_joint_ready_grasp = JointPosition(joints=[80.528153, 60.324139, -60.220359, 81.512108, 115.051359, -109.331694])
    # target_joint_grasp = JointPosition(joints=[80.528153, 60.075637, -65.493937, 87.034188, 115.051359, -109.331694])
    target_joint_ready_place = JointPosition(joints=[182.715424, 72.946464, -67.242515, 82.994, 90.0, -90.0])
    # target_joint_place = JointPosition(joints=[182.715424, 72.767592, -80.239649, 96.170006, 90, -90])
    # target_cartesian_place = CartesianPosition(xyzrxryrz=[-512.999982, -139.609811, 314.945566, -180.0, 1.302051, -177.284576])
    move_joint_grasp = jaka_zu7.move2joint_position_sync(target_joint_ready_grasp,jaka_move_parameters_joint)
    move_line_bias_response = move_line_bias(0,0,-30,jaka_move_parameters_cartesian)
    if move_line_bias_response:
        print("move_to_grasp_cartesian success")
        lmy.open(0)
        time.sleep(0.1)
        move_line_bias_response = move_line_bias(0,0,100,jaka_move_parameters_cartesian)
        if move_line_bias_response:
            print("grasp and hold on success")
            jaka_zu7.move2joint_position_sync(target_joint_ready_place,jaka_move_parameters_joint)
            print("ready to place")
            move_line_grasp_response = move_line_bias(0,0,-70,jaka_move_parameters_cartesian)
            if move_line_grasp_response:
                print("place success")
                lmy.open(10)
                time.sleep(0.1)
                move_line_home_response = move_line_bias(0,0,50,jaka_move_parameters_cartesian)
                if move_line_home_response:
                    print("ready to go back to home point")
                    jaka_zu7.move2joint_position_sync(home_joint_position,jaka_move_parameters_joint)    
                    print("complete task and move to home point succsee")   
                else:
                    raise ValueError("move_line faild")  
            else:
                raise ValueError("move_line faild")   
        else:
            raise ValueError("move_line faild")   
    else:
        raise ValueError("move_line faild")         
                
    