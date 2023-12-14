import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import copy

np.set_printoptions(suppress=True)
# # 加载点云
# pcd = o3d.io.read_triangle_mesh("/home/lxl/lmy/Sisyphe/point/RF_BYD.STL")
# tmp_pcd = pcd.sample_points_uniformly(1000000)
# # 点云AABB包围盒
# aabb = pcd.get_axis_aligned_bounding_box()
# aabb.color = (1, 0, 0)
# ab = aabb.get_center()

class Point:
    # def Trans(self,T):
    #     '''
    #     param:T:实际物体位姿[1x6]
    #     return:M:位置转换嘎矩阵[1x3]
    #     '''
    #     aabb = pcd.get_axis_aligned_bounding_box()
    #     aabb.color = (1, 0, 0)
    #     ab = aabb.get_center()
    #     x = T[0] - ab[0]
    #     y = T[1] - ab[1]
    #     z = T[2] - ab[2]
    #     M = np.array([x,y,z])
    #     return M
    def get_bbx_points(self,path="Product_FXH_001.stl"):
        pcd = o3d.io.read_triangle_mesh(path)
        aabb = pcd.get_axis_aligned_bounding_box()
        ab = aabb.get_center()
        x = aabb.get_extent()
        P1 = ab - 0.5*x
        P2 = ab + 0.5*x
        P_min = np.array([P1[0],P1[1],P1[2]])
        P_max = np.array([P2[0],P2[1],P2[2]])
        P = np.array([P_min,P_max])
        return P,ab

    def Trans(self,T,P1,P2):
        '''
        param:T:实际物体位姿[3x4]
        param:P1,P2,已知模型的抓取点
        return:P1_t,P2_t:实际求得的抓取点
        '''
        P,ab = self.get_bbx_points()
        transformation = np.identity(4)
        transformation[:3,:3] = T[:3,:3]
        transformation[0,3] = T[0,3] - ab[0]
        transformation[1,3] = T[1,3] - ab[1]
        transformation[2,3] = T[2,3] - ab[2]
        P1_m = np.array([P1[0],P1[1],P1[2],1]).T
        P2_m = np.array([P2[0],P2[1],P2[2],1]).T
        P1_d = transformation@P1_m
        P2_d = transformation@P2_m
        P1_t = P1_d[:3]
        P2_t = P2_d[:3]
        return P1_t,P2_t
    
    def getTcpPose(self,Rt1,Rt2,P1,P2):
        '''
        param:P1,P2:二指的两个抓取点
        param:Rt1:实际物体位姿[4*4]
        param:Rt2:末端到tcp转换矩阵[4*4]
        return:RT:4x4旋转矩阵
        '''
        #夹爪连线向量
        if P2[1] > P1[1]:
            v = P1 - P2
        else:
            v = P2 - P1

        y = Rt2[:3,1]
        # 两个向量
        Lx=np.sqrt(v.dot(v))
        Ly=np.sqrt(y.dot(y))
        #相当于勾股定理，求得斜线的长度
        cos_angle=v.dot(y)/(Lx*Ly)
        angle=np.arccos(cos_angle)
        T = np.array([np.cos(angle),-np.sin(angle),0,np.sin(angle),np.cos(angle),0,0,0,1]).reshape(3,3)
        Rt2[:3,:3] = Rt2[:3,:3]@T

        p = 0.5*(P1 + P2)

        RT_center = np.identity(4)
        RT_center[:3,:3] = Rt1[:3,:3]
        RT_center[0,3] = p[0]
        RT_center[1,3] = p[1]
        RT_center[2,3] = p[2]

        RT_tcp = RT_center @ Rt2
        print("带补偿的末端位姿:\n",RT_tcp)
        return RT_tcp

    def get_point3(self,P1,P2,P3,vec='z'):
        '''
        param:P1,P2,P3:输入三个人工标注抓取点
        return:P1,P2,P3:输出三个合适的抓取点
        '''
        if vec == 'z':
            P:float = (P1[2] + P2[2] + P3[2]) / 3
            P1[2] = P2[2] = P3[2] = P
        elif vec == 'y':
            P:float = (P1[1] + P2[1] + P3[1]) / 3
            P1[1] = P2[1] = P3[1] = P
        elif vec == 'x':
            P:float = (P1[0] + P2[0] + P3[0]) / 3
            P1[0] = P2[0] = P3[0] = P  
        return P1,P2,P3  

    def get_point2(self,P1,P2,vec='z'):
        '''
        param:P1,P2:输入两个人工标注抓取点
        return:P1,P2:输出两个合适的抓取点
        '''
        if vec == 'z':
            P:float = (P1[2] + P2[2]) / 2
            P1[2] = P2[2] = P
        elif vec == 'y':
            P:float = (P1[1] + P2[1]) / 2
            P1[1] = P2[1] = P
        elif vec == 'x':
            P:float = (P1[0] + P2[0]) / 2
            P1[0] = P2[0] = P  
        return P1,P2

    def get_circle(self,P1,P2,P3):
        '''
        param P1 P2 P3:空间不共线的三个点的坐标
        return Oc:拟合圆的坐标
        '''
        a1:float = P1[1]*P2[2] - P2[1]*P1[2] - P1[1]*P3[2] + P3[1]*P1[2] + P2[1]*P3[2] - P3[1]*P2[2]
        b1:float = -(P1[0]*P2[2] - P2[0]*P1[2] - P1[0]*P3[2] + P3[0]*P1[2] + P2[0]*P3[2] -P3[0]*P2[2])
        c1:float = P1[0]*P2[1] - P2[0]*P1[1] - P1[0]*P3[1] + P3[0]*P1[1] + P2[0]*P3[1] - P3[0]*P2[1]
        d1:float = -(P1[0]*P2[1]*P3[2] - P1[0]*P3[1]*P2[2] - P2[0]*P1[1]*P3[2] + P2[0]*P3[1]*P1[2] + P3[0]*P1[1]*P2[2] - P3[0]*P2[1]*P1[2])

        a2:float = 2*(P2[0] - P1[0])
        b2:float = 2*(P2[1] - P1[1])
        c2:float = 2*(P2[2] - P1[2])
        d2:float = P1[0]*P1[0] + P1[1]*P1[1] + P1[2]*P1[2] - P2[0]*P2[0] -P2[1]*P2[1] - P2[2]*P2[2]

        a3:float = 2*(P3[0] - P1[0])
        b3:float = 2*(P3[1] - P1[1])
        c3:float = 2*(P3[2] - P1[2])
        d3:float = P1[0]*P1[0] + P1[1]*P1[1] + P1[2]*P1[2] - P3[0]*P3[0] -P3[1]*P3[1] - P3[2]*P3[2]

        a:float = -(b1*c2*d3 - b1*c3*d2 - b2*c1*d3 + b2*c3*d1 + b3*c1*d2 - b3*c2*d1) / (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)
        b:float = (a1*c2*d3 - a1*c3*d2 - a2*c1*d3 + a2*c3*d1 + a3*c1*d2 - a3*c2*d1) / (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)
        c:float = -(a1*b2*d3 - a1*b3*d2 - a2*b1*d3 + a2*b3*d1 + a3*b1*d2 - a3*b2*d1) / (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)

        Oc = np.array([a,b,c])
        print("拟合圆心坐标Oc:\n",Oc)
        return Oc

    def get_Normal(self,Oc,P2,P3):
        '''
        param Oc P2 P3:空间不共线的三个点的坐标
        return Vec:过圆心的平面法向量
        '''
        OP2 = P2 - Oc
        OP3 = P3 - Oc
        q = OP2[1]*OP3[2] - OP3[1]*OP2[2]
        w = OP2[2]*OP3[0] - OP3[2]*OP2[0]
        e = np.fabs(OP2[0]*OP3[1] - OP3[0]*OP2[1])
        Vec = np.array([q,w,e])
        Vec = Vec / np.linalg.norm(Vec)
        print("法向量Vec:\n",Vec)
        return Vec
        
    def get_rotationMatrix(self,Oc,Vec,flag):
        '''
        param P1 :拟合圆上某个点的坐标
        Param Oc:拟合圆的坐标
        param Vec:空间平面法向量
        param flag:法兰盘中心与夹爪的距离:flag=1:315或flag=2:250
        return:RT:4x4旋转矩阵
        '''
        if flag == 1:
            d = 315
        elif flag == 2:
            d = 250
        #求末端tcp位置
        Ot = Oc + d*Vec / np.linalg.norm(Vec)
        print("末端tcp位置Ot:\n",Ot)

        #求末端tcp姿态
        x = np.array([1,0,0])
        z = -Vec
        y = np.cross(z,x)
        x = np.cross(y,z)
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        z = z / np.linalg.norm(z)

        RT = np.column_stack((x,y,z,Ot)) 
        RT = np.row_stack((RT, np.array([0,0,0,1])))
        print("转换矩阵RT:\n",RT)

        return RT
    
    def get_rotationMatrix1(self,Oc,Vec):
        '''
        Param Oc:拟合圆的坐标
        param Vec:空间平面法向量
        return:RT:4x4旋转矩阵
        '''
        #求末端tcp位置
        Ot = Oc
        # print("Ot:\n",Ot)

        #求末端tcp姿态
        x = np.array([1,0,0])
        z = Vec
        y = np.cross(z,x)
        x = np.cross(y,z)
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        z = z / np.linalg.norm(z)

        RT = np.column_stack((x,y,z,Ot)) 
        # print("RT:\n",RT)
        RT = np.row_stack((RT, np.array([0,0,0,1])))

        return RT


    def get_rotationMatrix_2p(self,P1,P2,flag,method='z'):
        '''
        param:P1,P2:二指的两个抓取点
        param:Vec
        param flag:法兰盘中心与夹爪的距离:flag=1:315或flag=2:250
        return:RT:4x4旋转矩阵
        '''
        Ro = np.asarray([[1,0,0],[0,1,0],[0,0,1]])
        if flag == 1:
            d = 300
        elif flag == 2:
            d = 250

        
        if method == 'z':
            Tz = np.array([1,0,0,0,np.cos(np.pi),-np.sin(np.pi),0,np.sin(np.pi),np.cos(np.pi)]).reshape(3,3)
            Tx = np.array([1,0,0,0,np.cos(-np.pi/6),-np.sin(-np.pi/6),0,np.sin(-np.pi/6),np.cos(-np.pi/6)]).reshape(3,3)
            # print("Rz",Rz[2])
            if P2[1] > P1[1]:
                v = P1 - P2
            else:
                v = P2 - P1
            
        elif method == 'y':
            Tz = np.array([1,0,0,0,np.cos(np.pi/2),-np.sin(np.pi/2),0,np.sin(np.pi/2),np.cos(np.pi/2)]).reshape(3,3)
            Tx = np.array([np.cos(np.pi/6),-np.sin(np.pi/6),0,np.sin(np.pi/6),np.cos(np.pi/6),0,0,0,1]).reshape(3,3)
            # print("Rz",Rz[2])
            if P2[2] > P1[2]:
                v = P2 - P1
            else:
                v = P1 - P2
        elif method == 'x':
            Tz = np.array([np.cos(-np.pi/2),0,np.sin(-np.pi/2),0,1,0,-np.sin(-np.pi/2),0,np.cos(-np.pi/2)]).reshape(3,3)
            Tx = np.array([np.cos(-np.pi/6),0,np.sin(-np.pi/6),0,1,0,-np.sin(-np.pi/6),0,np.cos(-np.pi/6)]).reshape(3,3)
            # print("Rz",Rz[2])
            if P2[2] > P1[2]:
                v = P2 - P1
            else:
                v = P1 - P2
        
        
        # Rm = Ro@Tx
        # Rz = Rm@Tz
        Rz = Ro@Tz
        Ro = Ro@Tz 
        #求末端tcp位置
        Ot = 0.5*(P1 + P2) - d*Rz[:,2] / np.linalg.norm(Rz[:,2])
        print("末端tcp位置Ot:\n",Ot)

        #求末端tcp姿态
        # 原始机械臂坐标系直接右乘x的旋转矩阵180度,再进行角度计算
        y = Ro[:,1]
        # 两个向量
        Lx=np.sqrt(v.dot(v))
        Ly=np.sqrt(y.dot(y))
        #相当于勾股定理，求得斜线的长度
        cos_angle=v.dot(y)/(Lx*Ly)
        #求得cos_sita的值再反过来计算，绝对长度乘以cos角度为矢量长度
        # print(cos_angle)
        angle=np.arccos(cos_angle)
    
        T = np.array([np.cos(angle),-np.sin(angle),0,np.sin(angle),np.cos(angle),0,0,0,1]).reshape(3,3)
        print("T:",T)
        M = Rz@T
        RT = np.column_stack((M,Ot)) 
        
        RT = np.row_stack((RT, np.array([0,0,0,1])))
        
        print("转换矩阵RT:\n",RT)

        return RT
    
    def get_rotationMatrix_2p1(self,P1,P2,method='z'):
        '''
        param:P1,P2:二指的两个抓取点
        param:Vec
        return:RT:4x4旋转矩阵
        '''
        if method == 'z':
            Vec = np.array([0,0,1])
            x = np.array([1,0,0])
        elif method == 'y':
            Vec = np.array([0,1,0])
            x = np.array([1,0,0])
        elif method == 'x':
            Vec = np.array([1,0,0])
            x = np.array([0,1,0])
        #求末端tcp位置
        Ot = 0.5*(P1 + P2)
        
        #求末端tcp姿态
        z = Vec
        y = np.cross(z,x)
        x = np.cross(y,z)
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        z = z / np.linalg.norm(z)

        RT = np.column_stack((x,y,z,Ot)) 
        # print("转换矩阵RT:\n",RT)
        RT = np.row_stack((RT, np.array([0,0,0,1])))

        return RT

    def xyz_rxryrz2transformation(self,xyz_rxryrz: np.ndarray): # only work for fannuc
        
        transformation = np.identity(4)
        transformation[:3, :3] = R.from_euler(seq="xyz", angles=xyz_rxryrz[3:], degrees=True).as_matrix()
        transformation[0, 3] = xyz_rxryrz[0]
        transformation[1, 3] = xyz_rxryrz[1]
        transformation[2, 3] = xyz_rxryrz[2]
        return transformation

    def transformation2xyz_rxryrz(self,transformation: np.ndarray):  # only work for fannuc
        rxryrz = R.from_matrix(transformation[:3, :3]).as_euler(seq="xyz", degrees=True)
        # for i in range(0,3): #确保角度不超过180
        #     if abs(180 - abs(rxryrz[i])) < 0.1:
        #         rxryrz[i] = 179.9
        return np.concatenate([transformation[:3,3],rxryrz])

if __name__== '__main__':
    
    # #三个抓取点
    # point = Point()
    # P1 = np.array([574.864929 ,9.698970,219.750107])
    # P2 = np.array([581.190430 ,-2.490505,217.971252])
    # P3 = np.array([572.031982 ,-10.554514,216.948761])
    # P1,P2,P3 = point.get_point3(P1,P2,P3)
    # Oc = point.get_circle(P1,P2,P3)
    # Vec = point.get_Normal(Oc,P2,P3)
    # RT = point.get_rotationMatrix(Oc,Vec,1)
    # RT2 = point.get_rotationMatrix1(Oc,Vec)
    # t = point.transformation2xyz_rxryrz(RT)
    # print("三指下机械臂tcp坐标为:\n",t)

    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame( size = 50)
    # mesh.compute_vertex_normals()
    # mesh_tcp = copy.deepcopy(mesh)
    # mesh_tcp.transform(RT2)
    

    # mesh_tcp1 = copy.deepcopy(mesh)
    # mesh_tcp1.transform(RT)

    # object_mesh_path = "RF_BYD.STL" 
    # object_mesh = o3d.io.read_triangle_mesh(object_mesh_path)
    # o3d.visualization.draw_geometries( [mesh_tcp1, mesh_tcp,object_mesh,mesh])

    #两个抓取点
    point = Point()
    P4 = np.array([571.610046 ,10.440644,222.961029])
    P5 = np.array([571.565735 ,-10.549973,219.105759])
    # P4 = np.array([558.200012 ,-15.647110,297.273956])
    # P5 = np.array([648.799988 ,-15.588098,296.637939])
    P4,P5 = point.get_point2(P4,P5)
    RT3 = point.get_rotationMatrix_2p(P4,P5,1,'z')
    RT4 = point.get_rotationMatrix_2p1(P4,P5,'z')
    t = point.transformation2xyz_rxryrz(RT3)
    print("二指下机械臂tcp坐标为:\n",t)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame( size = 50)
    mesh.compute_vertex_normals()
    mesh_tcp = copy.deepcopy(mesh)
    mesh_tcp.transform(RT4)
    

    mesh_tcp1 = copy.deepcopy(mesh)
    mesh_tcp1.transform(RT3)
    
    
    

    object_mesh_path = "RF_BYD.STL" 
    # object_mesh_path = "分线盒-1.STL"
    object_mesh = o3d.io.read_triangle_mesh(object_mesh_path)
    o3d.visualization.draw_geometries( [mesh_tcp1,mesh_tcp,object_mesh,mesh])