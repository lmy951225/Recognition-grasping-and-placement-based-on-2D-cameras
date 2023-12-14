# -*- coding: utf_8 -*-
import serial
import time
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu
from loguru import logger
class gripper_client():
    def __init__(self,port="/dev/ttyUSB0"):
        self.master = modbus_rtu.RtuMaster(serial.Serial(port=port, baudrate=115200, parity='N', stopbits=1, bytesize=8))
        self.port=port
        self.master.set_timeout(1.0)
        self.master.set_verbose(True)
        self.enable()
        self.set_force()

    def enable(self):
        self.master.execute(1, cst.WRITE_SINGLE_REGISTER, 0, output_value=1)
        print("Gripper at {} connected!".format(self.port))
    
    def close(self):
        self.master.close()
        
    def set_open_adjust(self): #张开校准，外抓
        self.master.execute(1,cst.WRITE_SINGLE_REGISTER,0x0082,output_value=0)
    
    def set_close_adjust(self): #闭合校准，内撑
        self.master.execute(1,cst.WRITE_SINGLE_REGISTER,0x0082,output_value=1)

    def open(self,pos=10.0):
        self.master.execute(1,cst.WRITE_MULTIPLE_REGISTERS,0x0002,data_format=">f",output_value=[pos])
    
    def set_force(self,force=0.5):
        self.master.execute(1,cst.WRITE_MULTIPLE_REGISTERS,0x0006,data_format=">f",output_value=[force])
    
    def get_position(self):
        return self.master.execute(1,cst.READ_HOLDING_REGISTERS,0x0042,2,data_format=">f")[0]
    
    def get_current(self):
        return self.master.execute(1,cst.READ_HOLDING_REGISTERS,0x0046,2,data_format=">f")[0]
    
    def get_State(self):
        return self.master.execute(1,cst.READ_HOLDING_REGISTERS,0x0041,1,data_format=">h")[0]
    
    def auto_init(self):
        self.master.execute(1,cst.WRITE_SINGLE_REGISTER,0x0083,output_value=0)
        self.master.execute(1,cst.WRITE_SINGLE_REGISTER,0x0084,output_value=1)


if __name__ == "__main__":
    # c1=gripper_client(port="/dev/ttyUSB2")
    # c1.enable()
    c2=gripper_client(port="/dev/ttyUSB0")
    c2.enable()
    print("starting")
    time.sleep(5)
    c2.set_force(0.3)
    # while True:
    #     print(c2.get_current())
    #     c2.open(0)
    #     # c1.open(c2.get_position())
    # c2.auto_init()
    # print("middle...")
    # c2.open(5)
    # time.sleep(5)
    print("close...")
    c2.open(0)
    time.sleep(5)
    print("open...")
    c2.open(10)
    time.sleep(5)