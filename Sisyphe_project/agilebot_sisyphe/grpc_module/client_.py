import os
import sys

from typing import List
import math
sys.path.append("..")
sys.path.append(os.getcwd())
base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
import grpc
import agile_robot_pb2 as pb2
import agile_robot_pb2_grpc as pb2_grpc
from Agilebot.IR.A import digital_signals

joints1 = pb2.JointPosition(
J1=5.235987755982989e-05,
J2=-1.7453292519943296e-05,
J3=-0.3491007569839058,
J4=-1.7453292519943296e-05,
J5=0.3490658503988659,
J6=0
)

joints2 = pb2.JointPosition(
  J1 = 0.6083345107826235,
  J2=-0.2557430952947291,
  J3=-0.21579250871657893,
  J4=0.7409446273491528,
  J5=0.8065864605166594,
  J6=-1.1503290566969429
)

joints3=  pb2.JointPosition(
      J1=0.5699547205312683,
  J2=-0.3974463772641487,
  J3=-0.26124088243851123,
  J4=0.5735675520828966,
  J5=0.8849517439312049,
  J6=-1.2509298347818958,
)


joints4 = pb2.JointPosition(
  J1=0.5039987280984025,
  J2=-0.5222548720742632,
  J3=-0.06733480254194124,
  J4= 0.5264436622790496,
  J5= 0.7980518004744072,
  J6= -1.2381365713647774
)
joints5 = pb2.JointPosition(
  J1=0.49801224876406197,
  J2= -0.6660874557311159,
  J3= 0.17385224679115518,
  J4= 0.917868653623818,
  J5= 0.8937132967762165,
  J6= -1.5856141721443284
)
joints6 =  pb2.JointPosition(
  J1= 0.4199436713223556,
  J2= -0.605716516904632,
  J3= 0.08740608893987602,
  J4= 0.904621604601181,
  J5= 0.828560155799268,
  J6= -1.2329180369013144
)
joints7 = pb2.JointPosition(
  J1= 0.27164304478039747,
  J2= -0.4979075290089423,
  J3= 0.018046704465621368,
  J4= 0.8484220026869634,
  J5= 0.6629807696625659,
  J6= -0.9022828634035086
)
joints8 =  pb2.JointPosition(
  J1=-0.09911724822075799,
  J2= -0.4549375228248419,
  J3= -0.05311036913818745,
  J4= -0.37124898519171384,
  J5= 0.5555383009097951,
  J6= 0.3500432347799827
)
joints9 = pb2.JointPosition(
  J1= -0.3816511475336001,
  J2= -0.5456073774659473,
  J3= 0.08806931405563387,
  J4= -0.8433430945636601,
  J5= 0.6590363255530588,
  J6= 0.7618711250805648
)
joints10 = pb2.JointPosition(
  J1= -0.6028192703463215,
  J2= -0.6472029932245373,
  J3= 0.24516640002764348,
  J4= -1.0624691821515482,
  J5= 0.8061501282036608,
  J6= 1.3163273218541234,
)
joints11 =  pb2.JointPosition(
  J1= -0.878755825086625,
  J2= -0.2697231826032037,
  J3= -0.32695252877609776,
  J4= -1.1504163231595423,
  J5= 1.1096977917105146,
  J6= 1.2382063845348572
)

joints = [joints1, joints2, joints3, joints4, joints5, joints6, joints7, joints8, joints9, joints10, joints11]


class Client():
    def __init__(self):
        self.conn = grpc.insecure_channel("0.0.0.0:49999")
        self.client = pb2_grpc.ControllerServerStub(channel=self.conn)

    def test_connect(self):
        req = pb2.ControllerIP(controller_ip="192.168.110.2")

        reply = self.client.connect(req)
        # print("test")
        print(reply.code.error_code)
        print(reply.log.res_log)

    def test_disConnect(self):
        req = pb2.FakeInput(fake_info="fake info")
        reply = self.client.disConnect(req)
        print(reply.code.error_code)
        print(reply.log.res_log)

    def test_reset(self): 
        req = pb2.FakeInput(fake_info="fake info")
        reply = self.client.reset(req)
        print(reply.code.error_code)
        print(reply.log.res_log)
    def test_getName(self):
        req = pb2.FakeInput(fake_info="fake info")
        reply = self.client.getName(req)
        print(reply.code.error_code)
        print(reply.log.res_log)
        print(reply.manipulator_type.manipulator_type)

    def test_enableRobot(self):
        req = pb2.FakeInput(fake_info="fake info")
        reply = self.client.enableRobot(req)
        print(reply.code.error_code)
        # print(reply.log.res_log)

    def test_disableRobot(self):
        req = pb2.FakeInput(fake_info="fake info")
        reply = self.client.disableRobot(req)
        print(reply.code.error_code)
        # print(reply.log.res_log)

    def test_setSpeedRatio(self):
        req = pb2.SetSpeedRatioRequest(speed_ratio=0.1)
        reply = self.client.setSpeedRatio(req)
        print(reply.code.error_code)
        print(reply.log.res_log)

    def test_getIO(self):
        req = pb2.GetIORequest(io_channels=[1,2],io_type=2)
        reply = self.client.getIO(req)
        print(reply.code.error_code)
        print(reply.log.res_log)


    def test_setIO(self):
        req = []

        for i in range(2):
            req.append(pb2.IOState(io=1, state=1, io_type=2))
        reply = self.client.setIO(pb2.SetIORequest(io_states=req))

        print(reply.code.error_code)
        print(reply.log.res_log)
        # print(reply.log)
        # print(reply.io_states[0])

    def test_getJointPosition(self):
        req = pb2.FakeInput(fake_info="fake info")
        reply = self.client.getJointPosition(req)

        print(reply.code.error_code)
        print(reply.log.res_log)

        print(f'J1 {reply.joints.J1}')
        return reply.joints.J1, reply.joints.J2, reply.joints.J3, reply.joints.J4, reply.joints.J5, reply.joints.J6

    def test_getPose(self):
        req = pb2.FakeInput(fake_info="fake info")
        reply = self.client.getPose(req)

        print(reply.code.error_code)
        print(reply.log.res_log)

        print(f'x {reply.pose.x}')
        return reply.pose.x, reply.pose.y, reply.pose.z, reply.pose.rx, reply.pose.ry, reply.pose.rz

    def test_moveJoint(self):
        J1, J2, J3, J4, J5, J6 = self.test_getJointPosition()

        req = pb2.JointPosition(J1=J1,
                                J2=J2,
                                J3=J3 + math.radians(5),
                                J4=J4,
                                J5=J5,
                                J6=J6)

        reply = self.client.moveJoint(req)
        print(reply.code.error_code)
        print(reply.log.res_log)
    
    def test_moveJoint_data(self,joints):


        # data = joints['J1']

        # print(f'J1 {data}')

        req = pb2.JointPosition(J1=joints['J1'],
                                J2=joints['J2'],
                                J3=joints['J3'],
                                J4=joints['J4'],
                                J5=joints['J5'],
                                J6=joints['J6'])

        reply = self.client.moveJoint(req)

        print(reply.code.error_code)
        print(reply.log.res_log)

    def test_movePose(self):
        x, y, z, a, b, c = self.test_getPose()
        print(x, y, z, a, b, c)
        req = pb2.CartesianPosition(x=x,
                                    y=y,
                                    z=z + 50,
                                    rx=a,
                                    ry=b,
                                    rz=c)
        reply = self.client.movePose(req)
        print(reply.code.error_code)
        print(reply.log.res_log)
        x, y, z, a, b, c = self.test_getPose()
        print(x, y, z, a, b, c)



    def test_uploadFlyShotTraj(self):
        request: pb2.UploadFlyShotTrajRequest = pb2.UploadFlyShotTrajRequest(name="test_trajectory", 
                                                                            joints=[pb2.JointPosition(J1=0, J2=0, J3=0, J4=0, J5=0, J6=0),
                                                                                    pb2.JointPosition(J1=0, J2=0, J3=0, J4=0, J5=0, J6=0),
                                                                                    pb2.JointPosition(J1=0, J2=0, J3=0, J4=0, J5=0, J6=0),
                                                                                    pb2.JointPosition(J1=0, J2=0, J3=0, J4=0, J5=0, J6=0),
                                                                                    pb2.JointPosition(J1=0, J2=0, J3=0, J4=0, J5=0, J6=0)],
                                                                            shot_flags=[True, False, True, False, False],
                                                                            offset=[0, 0, 0, 0, 0], 
                                                                            traj_addrs=[pb2.IOAddrs(addrs=[0, 1]),
                                                                                        pb2.IOAddrs(addrs=[1, 2]),
                                                                                        pb2.IOAddrs(addrs=[3, 4]),
                                                                                        pb2.IOAddrs(addrs=[4, 5]),
                                                                                        pb2.IOAddrs(addrs=[5, 6])])
        reply = self.client.uploadFlyShotTraj(request)
        print(reply)
    
    def test_getRobotProgramInfo(self):
        req = pb2.GetRobotProgramRequest(name="test0518")
        reply = self.client.getRobotProgramInfo(req)

        print(reply.code.error_code)
        print(reply.log.res_log)
        
    def test_isFlyShotTrajValid(self):
        
        request:pb2.IsFlyShotTrajValidRequest = pb2.IsFlyShotTrajValidRequest(
            name="test0",
            method="TOPPRA",
            save_traj=True,
            kinematics_params=pb2.kinematicsParams(
                joint_vel_coef=[0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                joint_acc_coef=[0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                joint_jerk_coef=[0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
            )
        )
        reply = self.client.isFlyShotTrajValid(request)
        print(reply)
        
    
    def test_optimizeKinematicsParams(self):
        
        request:pb2.OptimizeKinematicsParamsRequest = pb2.OptimizeKinematicsParamsRequest(
            name="test0",
            method="TOPPRA"
        )
        reply = self.client.optimizeKinematicsParams(request)
        print(reply)        
        
    
    def test_executeFlyShotTraj(self):
        
        request:pb2.ExecuteFlyShotTrajRequest = pb2.ExecuteFlyShotTrajRequest(
            name="test0",
            method="TOPPRA",
            move_to_start=True,
            use_cache=False,
            kinematics_params=pb2.kinematicsParams(
                joint_vel_coef=[0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                joint_acc_coef=[0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                joint_jerk_coef=[0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
            )
        )
        reply = self.client.executeFlyShotTraj(request)
        print(reply)
        
    def test_uploadFlyShotTraj(self):
        request: pb2.UploadFlyShotTrajRequest = pb2.UploadFlyShotTrajRequest(name="test0", 
                                                                        joints=[joints1, joints2, joints3, joints4, joints5, joints6, joints7, joints8, joints9, joints10, joints11],
                                                                        shot_flags=[True] * 11,
                                                                        offset=[0] * 11,
                                                                        traj_addrs = [pb2.IOAddrs(addrs=[1,2,3,4,5,6,7,8])] * 11)
        print(self.client.uploadFlyShotTraj(request))

    def test_deleteFlyShotTraj(self):
        request: pb2.DeleteFlyShotTrajRequest = pb2.DeleteFlyShotTrajRequest(name="test0")
        print(self.client.deleteFlyShotTraj(request))

    def test_movePose_grasp(self):
        x, y, z, a, b, c = self.test_getPose()
        print(x, y, z, a, b, c)
        req = pb2.CartesianPosition(x=571.5878905 ,
                                    y=-0.0546645,
                                    z=536.033394,
                                    rx=179.9,
                                    ry=0,
                                    rz=0)
        reply = self.client.movePose(req)
        print(reply.code.error_code)
        print(reply.log.res_log)
        x, y, z, a, b, c = self.test_getPose()
        print(x, y, z, a, b, c)

        # [571.5878905  -0.0546645 536.033394  179.9         0.          0.       ]


    


if __name__ == '__main__':
    c = Client()
    c.test_connect()
    #c.test_setSpeedRatio()
    # c.test_disableRobot()
    # c.test_getPose()
    # c.test_reset()
    # c.test_enableRobot()
    # c.test_enableRobot()
    # c.test_disConnect()

    # c.test_setSpeedRatio()
    # c.test_setIO()
    # c.test_getJointPosition()
    # c.test_moveJoint()
    c.test_movePose()
    # c.test_movePose_grasp()
    # c.test_getRobotProgramInfo()
    # c.test_uploadFlyShotTraj()
    # #c.test_moveJoint_data(joints=joints1)
    # # c.test_connect()
    # #c.test_moveJoint_data(joints=joints2)
    # # c.test_connect()
    # #c.test_moveJoint_data(joints=joints3)
    # #c.test_uploadFlyShotTraj()
    # c.test_isFlyShotTrajValid()
    # c.test_optimizeKinematicsParams()
    # c.test_disConnect()

