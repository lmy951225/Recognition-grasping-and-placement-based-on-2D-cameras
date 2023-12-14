import open3d as o3d
import os
import numpy as np
import copy
from utils import get_circle
np.set_printoptions(suppress=True)
from utils_rot import rotationMatrixToEulerAngles


mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
mesh_ico=o3d.geometry.TriangleMesh.create_icosahedron(radius=1.0)


pcds0 = o3d.geometry.PointCloud()
pcds0.points=mesh_ico.vertices
# o3d.visualization.draw_geometries([mesh,pcds0])

mesh_ico_2 = mesh_ico.subdivide_loop(number_of_iterations=4)
print(
    f'After subdivision it has {len(mesh_ico_2.vertices)} vertices and {len(mesh_ico_2.triangles)} triangles'
)
# o3d.visualization.draw_geometries([mesh_ico_2,mesh], mesh_show_wireframe=True)

pcds0.points=mesh_ico_2.vertices
# o3d.visualization.draw_geometries([pcds0,mesh])#, mesh_show_wireframe=True)


points=np.asarray(pcds0.points)
triangles=np.asarray(mesh_ico_2.triangles)



# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
#
#
# fig=plt.figure()
# ax = fig.add_subplot(projection='3d')
#
# points_tri=points[list(triangles[100])]#list(triangles[0])+
# ax.plot(points_tri[:,0],points_tri[:,1],points_tri[:,2],'.')
# ax.set_xlim([-2,2])
# ax.set_ylim([-2,2])
# ax.set_zlim([-2,2])

##
points_tri_0=points[triangles[0]]
points_tri_100=points[triangles[100]]
##length to origin:
# np.linalg.norm(points_tri_0,axis=-1)

## three points of triangle
p1,p2,p3=points_tri_100[0],points_tri_100[1],points_tri_100[2]
theta_12=180/np.pi*np.arccos(np.dot(p1,p2)/(np.linalg.norm(p1)*np.linalg.norm(p2)))
theta_13=180/np.pi*np.arccos(np.dot(p1,p3)/(np.linalg.norm(p1)*np.linalg.norm(p3)))
theta_23=180/np.pi*np.arccos(np.dot(p2,p3)/(np.linalg.norm(p2)*np.linalg.norm(p3)))
print(theta_12,theta_13,theta_23)
############

## get orient
def get_orient(p1):
    vector_z=np.array([0,0,-1])##
    # cam_z
    vector_cam_z=-p1/np.linalg.norm(-p1)##
    # cam_x
    vector_cam_x=np.cross(vector_cam_z,vector_z)
    vector_cam_x=vector_cam_x/np.linalg.norm(vector_cam_x)
    # cam_y
    vector_cam_y=np.cross(vector_cam_z,vector_cam_x)
    vector_cam_y=vector_cam_y/np.linalg.norm(vector_cam_y)
    ##
    R_cam=np.concatenate([vector_cam_x[:,None],vector_cam_y[:,None],vector_cam_z[:,None]],axis=-1)
    t_cam=p1.copy()
    T_cam=np.concatenate([np.concatenate([R_cam,t_cam[:,None]],axis=-1),np.array([[0,0,0,1]])],axis=0)
    #
    mesh_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    mesh_cam.transform(T_cam)
    return T_cam,mesh_cam

# o3d.visualization.draw_geometries([pcds0,mesh])#, mesh_show_wireframe=True)
# vis_list=[pcds0,mesh]
# for idx,p in enumerate(points):
#     _,mesh_cam=get_orient(p)
#     vis_list.append(mesh_cam)
#     print(idx)
#
# o3d.visualization.draw_geometries(vis_list)#, mesh_show_wireframe=True)

######## filter 60 degree:
vis_list=[]
points_filter=[]
angle_up_list=[]
angle_plane_list=[]
for idx,p in enumerate(points):
    T_cam,mesh_cam=get_orient(p)
    ##
    vector_cam_z=T_cam[:3,2]
    theta=180/np.pi*np.arccos(np.dot(vector_cam_z,np.array([0,0,-1]))/np.linalg.norm(vector_cam_z))
    if 90-theta<55:#30:
        continue
    vector_cam_z_proj=vector_cam_z.copy()
    vector_cam_z_proj[-1]=0
    theta_plane=180/np.pi*np.arccos(np.dot(vector_cam_z_proj,np.array([1,0,0]))/np.linalg.norm(vector_cam_z_proj))
    if vector_cam_z_proj[1]<0:
        theta_plane*=-1
    ############ 渐变半径
    ratio=0.7#1.5#  0<ratio<=3
    y_start=30 / 180 * ratio
    y_end=90/180
    a=(y_end-y_start)/(60/180)
    b=0.5-0.5*a
    if 90-theta<float('inf'):
        mesh_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        ##
        x=(90-theta)/180
        y=a*x+b
        ##
        p *= np.sin(np.pi*y)
        T_cam[:3,-1]=p
        mesh_cam.transform(T_cam)
    #######
    angle_up_list.append(90 - theta)
    angle_plane_list.append(theta_plane)
    ###
    points_filter.append(p[None,:])
    vis_list.append(mesh_cam)
    print(idx)

#####
points_filter=np.concatenate(points_filter,axis=0)
print('points_filter',points_filter.shape)

angle_up_arr=np.array(angle_up_list)
up_region_list=[[55,60],[60,65],[65,70],[70,75],[75,80],[80,85],[85,90]]
region_index_list=[]
for up_region in up_region_list:
    low,high=up_region[0],up_region[1]
    index=np.where((angle_up_arr>low)&(angle_up_arr<=high))[0]
    print(index.shape)
    region_index_list.append(index)

points_filter_sort=[]
angle_plane_arr=np.array(angle_plane_list)
for region_index in region_index_list:
    region_points=points_filter[region_index]
    region_angle_plane_arr=angle_plane_arr[region_index]
    index_plane=np.argsort(region_angle_plane_arr)
    points_filter_sort.append(region_points[index_plane])


points_filter_sort=np.concatenate(points_filter_sort,axis=0)



# pcds0_filter = o3d.geometry.PointCloud()
# pcds0_filter.points = o3d.utility.Vector3dVector(points_filter)
# vis_list.append(mesh)
# vis_list.append(pcds0_filter)
# o3d.visualization.draw_geometries(vis_list)#, mesh_show_wireframe=True)


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

#######
fig=plt.figure()
#######
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
for p in points_filter_sort:
    ax.scatter(p[0],p[1],p[2])
    plt.pause(0.01)





