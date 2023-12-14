from typing import List, Tuple
import struct
import socket
import logging
import copy
import math
import time
import pandas
import numpy as np
import numpy.typing as npt
from RobotDriverCore.RobotDriverCoreUtils.RobotPosition import JointPosition, CartesianPosition
from RobotDriverCore.RobotDriverCoreUtils.SignalIO import SignalIOType
from RobotDriverCore.RobotDriverCoreUtils.RobotConstraints import RobotConstraints

class FanucStreamMotionJointPosition(JointPosition):
    def __init__(self, joints: List[float]) -> None:
        super().__init__(joints)
        self._j7 = 0
        self._j8 = 0
        self._j9 = 0
 
    def to_bytes(self) -> bytes:
        return struct.pack('!fffffffff', self._j1, self._j2, self._j3, self._j4, self._j5, self._j6,self._j7, self._j8, self._j9)
    
class FanucStreamMotionCartesianPosition(CartesianPosition):
    def __init__(self, xyzrxryrz: List[float]) -> None:
        super().__init__(xyzrxryrz)
        self._extend_axis1 = 0
        self._extend_axis2 = 0
        self._extend_axis3 = 0

class IOinfo():
    def __init__(self, io_type: int, io_index: int, io_mask: int = 0, io_value: int = -1) -> None:
        # io_type 1 byte unsigned int
        # io_index 2 bytes unsigned int
        # io_mask 2 byte unsigned int
        # io_value 2 byte unsigned int
        self.__io_type = io_type
        self.__io_index = io_index
        self.__io_mask = io_mask
        self.__io_value = io_value
    
    def to_bytes(self) -> bytes:
        tmp_bytes = struct.pack("!BHH", self.__io_type, self.__io_index, self.__io_mask)
        if self.__io_value < 0:
            return tmp_bytes
        else:
            return tmp_bytes + struct.pack('!H', self.__io_value)

novalue_io_info_bytes = IOinfo(io_type=SignalIOType.SIGNAL_NO_IO.Fanuc_code, io_index=0, io_mask=0).to_bytes()
value0_io_info= IOinfo(io_type=SignalIOType.SIGNAL_NO_IO.Fanuc_code, io_index=0, io_mask=0, io_value=0)

class JointsCommandPacket():
    def __init__(self, sequence_no: int, is_last: bool, write_io_info: IOinfo, joint_position: FanucStreamMotionJointPosition) -> None:
        self.__sequence_no = sequence_no
        self.__is_last = is_last
        self.__write_io_info: IOinfo = write_io_info
        self.__joint_position: FanucStreamMotionJointPosition = joint_position
    
    def to_bytes(self):
        result: bytes = struct.pack("!IIIB", 1, 1, self.__sequence_no, self.__is_last) + novalue_io_info_bytes + struct.pack('!B', 1)\
                        + self.__write_io_info.to_bytes() + struct.pack('!H', 0) + self.__joint_position.to_bytes()
        return result

INIT_PACKET = struct.pack('!II', 0, 1)
END_PACK = struct.pack('!II', 2, 1)

def csv_row2cmd_bytes(row: pandas.Series, sequenceNo: int, is_last:bool = False) -> bytes:
    joints_dgree = [math.degrees(row.iloc[j]) for j in range(1, 7)]
    cmd_joint_position = FanucStreamMotionJointPosition(joints=joints_dgree)
    if row.iloc[25] not in ['-1', -1]: # compatible with zero
        io_ports = [int(float(ch)) for ch in str(row.iloc[25]).split('|')]
        first_io_index = min(io_ports)
        io_masks = sum([pow(2, int(i - first_io_index)) for i in io_ports])
        if row.iloc[26] == 0:
            io_values = 0
        else:
            io_values = io_masks
        write_io_info = IOinfo(io_type=SignalIOType.SIGNAL_DO.Fanuc_code, io_index=first_io_index, io_mask=io_masks, io_value=io_values)
    else:
        write_io_info = value0_io_info
    cmd_packet = JointsCommandPacket(sequence_no=sequenceNo, is_last=is_last, write_io_info=write_io_info, joint_position=cmd_joint_position)
    return cmd_packet.to_bytes()

def delay_6_step(original_trajectory_csv: pandas.DataFrame) -> pandas.DataFrame:
    """"In the case of an robot with 8ms interval, there is a delay of
        about 50 ms including communication delay and servo tracking
        delay. (It becomes about half of that in case of a robot with 4ms
        interval) There is no way to make this shorter." 
        quote from stream motion(J519) document

    Args:
        original_trajectory_csv (pandas.DataFrame): original trajectory without delay

    Returns:
        pandas.DataFrame: delayed trajectory
    """
    original_indices = original_trajectory_csv.loc[(original_trajectory_csv['do_port'] != '-1') & (original_trajectory_csv['do_port'] != -1)].index.tolist()
    new_indices = [i + 6 for i in original_indices]
    if len(new_indices) == 0:
        return original_trajectory_csv
    # check only last one
    if new_indices[-1] > len(original_trajectory_csv) - 1:
        delta_step = new_indices[-1] - new_indices[-2]
        new_indices[-1] = len(original_trajectory_csv) - 1
        new_indices[-2] = new_indices[-1] - delta_step
    # assign old index content to new one
    new_trajectory = copy.deepcopy(original_trajectory_csv)
    new_trajectory.loc[:, 'do_port'] = '-1'
    new_trajectory.loc[:, 'do_state'] = 0
    for old_index, new_index in zip(original_indices, new_indices):
        new_trajectory.loc[new_index, ['do_port', 'do_state']] = original_trajectory_csv.loc[old_index, ['do_port', 'do_state']]
    return new_trajectory

def FanucJ519_stream_motion_prepare(trajectory_csv: pandas.DataFrame) -> List[bytes]:
    results: List[bytes] = list()
    sequence_no = 0
    delayed_trajectory = delay_6_step(trajectory_csv)
    for index, row in delayed_trajectory.iterrows():
        is_last = False if index < len(delayed_trajectory) - 1 else True
        sequence_no += 1
        results.append(csv_row2cmd_bytes(row, sequenceNo=sequence_no, is_last=is_last))
    return results

class SocketUDP():
    def __init__(self, robot_ip: str) -> None:
        self.__s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.__addr = (robot_ip, 60015)
        self.__s.settimeout(1)
    
    def send_message(self, data: bytes, receive_data: bool = True) -> Tuple[int, int]:
        self.__s.sendto(data, self.__addr)
        sequence_no = data_status = 0
        if receive_data:
            data = self.__s.recv(132)
            sequence_no, data_status = struct.unpack("!IB", data[8:13])
        return sequence_no, data_status

def FanucJ519_stream_motion_execute(controller_ip, executable_trajectory_path: str) -> bool:
    s = SocketUDP(robot_ip=controller_ip)
    with open(executable_trajectory_path, 'rb') as f:
        cmd_byte = f.read(64)
        sequence_no, data_status = s.send_message(INIT_PACKET)
        if (sequence_no == 1) and (data_status & 1):
            logging.info("succesful start init stream motion, start to move")
            # do normal receive and send
            while (cmd_byte):
                sequence_no, data_status = s.send_message(cmd_byte)
                if not (data_status & 1):
                    logging.error("wrong data status {} at sequence Number {}".format(data_status, sequence_no))
                    return False
                cmd_byte = f.read(64)
        else:
            logging.info("fail to init stream motion, please check")
            return False
    s.send_message(END_PACK, receive_data=False)
    return True

if __name__ == '__main__':
    FanucJ519_stream_motion_prepare(pandas.read_csv('11-1-1.csv'))
