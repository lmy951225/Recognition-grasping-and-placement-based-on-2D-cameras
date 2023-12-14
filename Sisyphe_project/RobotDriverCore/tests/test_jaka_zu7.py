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

#实例化对象
jaka_zu7 = JAKA_Zu7()
#连接
assert jaka_zu7.connect(controller_ip="192.168.13.133")
#上电
jaka_zu7.power_on()
#设置速度
jaka_zu7.set_speed_ratio(speed_ratio=0.2)

#获取当前关节角度
current_joint_position = jaka_zu7.get_current_joint_position()
#获取当前tcp
current_cartesian_position = jaka_zu7.get_current_cartesian_position()
print("current_joint_positon {}".format([current_joint_position.j1,current_joint_position.j2,current_joint_position.j3
                                         ,current_joint_position.j4,current_joint_position.j5,current_joint_position.j6]))
print("current_cartesian_positon {}".format([current_cartesian_position.x, current_cartesian_position.y, current_cartesian_position.z,
                                            current_cartesian_position.rx, current_cartesian_position.ry, current_cartesian_position.rz]))

#正运动学
fk = jaka_zu7.forward_kinematics(joint_position=current_joint_position)

#逆运动学
# target_cartesian_position = CartesianPosition(xyzrxryrz=[current_cartesian_position.x, current_cartesian_position.y, current_cartesian_position.z - 100,
#                                                          current_cartesian_position.rx, current_cartesian_position.ry, current_cartesian_position.rz])
target_cartesian_position = CartesianPosition(xyzrxryrz=[-110.208404, 329.503461, 576.22938,
                                                         180,0,-90])
ik = jaka_zu7.inverse_kinematics(cartesian_position=target_cartesian_position,reference_joint_position=current_joint_position)
if isinstance(ik,JointPosition):
    print("ik joint_positon {}".format([ik.j1,ik.j2,ik.j3,ik.j4,ik.j5,ik.j6]))
    
else:
    print("逆解失败")

#上使能
jaka_zu7.enable() 
#time.sleep(1) # wait between disable and enable
#下使能    
#jaka_zu7.disable()

#获取DH参数
# jaka_zu7.get_DH_parameters()

#清除错误
# jaka_zu7.reset()

jaka_move_parameters = {
    "total_wait_time":100,
    # "speed":50,
    # "accel":50,
    # "tol":0.2,
}
#move_l  #只要直线上存在奇异点就不会运动(哪怕终点可达)
# move_cartesian_response = jaka_zu7.move_line_sync(position=target_cartesian_position, jaka_move_parameters=jaka_move_parameters)
# print("move line finish {}".format(move_cartesian_response.value))
# if move_cartesian_response.code != 0:
#     print("move_line faild")


# print(jaka_zu7.JAKA_robot_state.raw_data) #打印10000端口所有状态

# #move_j(可输入关节值或者笛卡尔值，但是不走直线)
# target_joint_position = target_cartesian_position
# move_joint_response = jaka_zu7.move2joint_position_sync(joint_position=target_joint_position, jaka_move_parameters=jaka_move_parameters)
# print("move joint position finish {}".format(move_joint_response.value))
# if move_joint_response.code != 0:
#     print("move_joint faild")

# # # #move to init_point
# # target_joint_position = JointPosition(joints=[0,90,0,90,180,0]) #home_point,使用该点作为初始move_l路径会存在奇异
target_joint_position = JointPosition(joints=[91.144379, 92.363768, -17.412272, 104.48391, 91.960157, -90.006056]) #预设点
move_joint_response = jaka_zu7.move2joint_position_sync(joint_position=target_joint_position, jaka_move_parameters=jaka_move_parameters)
print("move joint position finish {}".format(move_joint_response.value))
if move_joint_response.code != 0:
    print("move_joint faild")