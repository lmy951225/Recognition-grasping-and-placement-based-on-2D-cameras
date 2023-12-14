from pyModbusTCP.client import ModbusClient
from loguru import logger
import os
add_mode = 0
add_position = 10
add_srength = 11
add_speed = 12
add_acc = 13
add_tongbu = 42
add_read_strength = 18

class ZXHand:
    client = None
    def __init__(self,ip = "192.168.2.90"):
        self.client = ModbusClient(host=ip,port=502,auto_open=True,auto_close=True)

        self.client.write_single_register(add_mode,1)
        self.client.write_single_register(add_tongbu,1)
        self.client.write_single_register(add_position,50)
        self.client.write_single_register(add_srength,100)
        self.client.write_single_register(add_speed,100)
        self.client.write_single_register(add_acc,50)

    def close(self,pos=0,speed=100,strength=300):
        self.client.write_single_register(add_position,pos)
        self.client.write_single_register(add_srength,strength)
        self.client.write_multiple_coils(add_speed,speed)

    def open(self,pos=100,speed=100,strength=300):
        self.client.write_single_register(add_position,pos)
        self.client.write_single_register(add_srength,strength)
        self.client.write_multiple_coils(add_speed,speed)

    def get_strength(self):
        logger.info("The current is {}".format(self.client.read_holding_registers(add_read_strength)[0]))
        return self.client.read_holding_registers(add_read_strength)[0]
    
if __name__== '__main__':
    logDir = os.path.expanduser('logs2')  # expanduser函数，它可以将参数中开头部分的 ~ 或 ~user 替换为当前用户的home目录并返回
    # 按照时间命名
    # logFile = os.path.join(logDir, 'current_log_{time}.log')
    logFile = os.path.join(logDir, 'current_log_{time}.log')
    if not os.path.exists(logDir):
        os.mkdir(logDir)

    logger.add(logFile)
    lmy = ZXHand()
    while True:
        lmy.close(strength=100)
        # lmy.close(strength=200)
        # lmy.close(strength=300)
        # lmy.close(strength=400)
        # lmy.open()
        lmy.get_strength()
    
        