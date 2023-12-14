from ruckig import InputParameter, OutputParameter, Result, Ruckig
from copy import copy 
import numpy as np 
import sys
import os
sys.path.append("..")
sys.path.append(os.getcwd())
base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
import matplotlib.pyplot as plt 
from Pose.IK import getJoint
from RobotDriverCore.ManipulatorDriver.AgilebotDriver.AgilebotDriver import AgilebotDriver
import pandas as pd
np.set_printoptions(suppress=True)

def walk_through_trajectory(otg, inp):
    out_list = []
    out = OutputParameter(inp.degrees_of_freedom)

    res = Result.Working
    while res == Result.Working:
        res = otg.update(inp, out)
        # print('\t'.join([f'{out.time:0.3f}'] + [f'{p:0.3f}' for p in out.new_position]))
        out.pass_to_input(inp)
        out_list.append(copy(out))

    return out_list

def multi_dimensional_p2p_ruckig(
    current_positions, 
    target_position, 
    current_velocity, 
    target_velocity, 
    current_acceleration, 
    target_acceleration, 
    v_boundary,
    a_boundary,
    j_boundary,
    # deltaT,
    T_s
):
    inp = InputParameter(6)
    inp.current_position = current_positions
    inp.target_position = target_position
    inp.current_velocity = current_velocity
    inp.target_velocity = target_velocity
    inp.current_acceleration = current_acceleration
    inp.target_acceleration = target_acceleration
    
    inp.max_velocity = v_boundary[1]
    inp.max_acceleration = a_boundary[1]
    inp.max_jerk = j_boundary[1]


    otg = Ruckig(inp.degrees_of_freedom, T_s)

    out_list = walk_through_trajectory(otg, inp)
    
    T = list(map(lambda x: x.time, out_list)) 

    Q = np.array(list(map(lambda x: x.new_position, out_list)))
    V = np.array(list(map(lambda x: x.new_velocity, out_list)))
    A = np.array(list(map(lambda x: x.new_acceleration, out_list)))
    J = np.diff(A, axis=0, prepend=A[0, 0]) / otg.delta_time
    duration = out_list[0].trajectory.duration
    inter_time = [T_s] + out_list[0].trajectory.intermediate_durations
    J[0, :] = 0.0
    J[-1, :] = 0.0
    
    # error = max([abs(Q[-1] - q1) , abs(V[-1] - self.sigma *self.v1), abs(A[-1] - self.sigma *self.a1)])
    
    for j in range(len(current_positions)):
        
        
        q1 = target_position[j]
        v1 = target_velocity[j]
        
        a1 = target_acceleration[j]
        
        Q_ = Q.tolist()[-1]
        V_ = V.tolist()[-1]
        A_ = A.tolist()[-1]
        # print(Q[-1])
        error = max([abs(Q[-1][j] - q1) , abs(V[-1][j] - v1), abs(A[-1][j] - a1)])
        # if error > 0.1:
            
            
        #     print([abs(Q[-1][j] - q1) , abs(V[-1][j] - v1), abs(A[-1][j] - a1)])
        #     raise(ValueError)
     
    
    return T, Q.T.tolist(), V.T.tolist(), A.T.tolist(), J.T.tolist(), duration,inter_time

def multi_dimensional_ruckig(
    thetas_list: np.ndarray, 
    v_boundary: np.ndarray, 
    a_boundary: np.ndarray, 
    j_boundary: np.ndarray, 
    T_s: float
):
    dof = len(thetas_list.T)
    # cycle_time = T_s

    # df = pts[cols].copy()
    wp_num = len(thetas_list)

    otg = Ruckig(dof, T_s, wp_num)  # DoFs, control cycle rate, maximum number of intermediate waypoints for memory allocation
    inp = InputParameter(dof)  # DoFs
    out = OutputParameter(dof, wp_num)  # DoFs, maximum number of intermediate waypoints for memory allocation

    inp.current_velocity = [0] * dof
    inp.current_acceleration = [0] * dof
    inp.target_velocity = [0] * dof
    inp.target_acceleration = [0] * dof
    
    wp_list = []
    for i, thetas in enumerate(thetas_list):
        if i == 0:
            inp.current_position = list(thetas)
            
        elif i == wp_num - 1:
            inp.target_position = list(thetas)
        else:
            wp_list.append(list(thetas))
            
    inp.intermediate_positions = wp_list.copy()   
    
    inp.max_velocity = v_boundary[1,:]
    inp.max_acceleration = a_boundary[1,:]
    inp.max_jerk = j_boundary[1,:]
    
    first_output, out_list = None, []
    res = Result.Working
    
    while res == Result.Working:
        res = otg.update(inp, out)

        print('\t'.join([f'{out.time:0.3f}'] + [f'{p:0.3f}' for p in out.new_position]))
        out_list.append(copy(out))

        out.pass_to_input(inp)

        if not first_output:
            first_output = copy(out)
            
    T = list(map(lambda x: x.time, out_list)) 
    
    
    Q = np.array(list(map(lambda x: x.new_position, out_list)))
    V = np.array(list(map(lambda x: x.new_velocity, out_list)))
    A = np.array(list(map(lambda x: x.new_acceleration, out_list)))
    J = np.diff(A, axis=0, prepend=A[0, 0]) / otg.delta_time
    
    duration = first_output.trajectory.duration
    inter_time = [T_s] + first_output.trajectory.intermediate_durations
    J[0, :] = 0.0
    J[-1, :] = 0.0
    
    # inter_time = []
    # for thetas in thetas_list:
        
    #     dist = np.sum((thetas - Q) ** 2, axis=1)
    
    #     index = np.argmin(dist)
    #     inter_time.append(T[index])
        
    # inter_time[-1] = T[-1]
    # inter_time[0] = T[0]
    # print(out_list)
    return T, Q.T.tolist(), V.T.tolist(), A.T.tolist(), J.T.tolist(), duration, inter_time

def draw_graph(T, Q, V, A, J, interval_time, q_list, leg):
        
    fs = 15
  
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.subplot(4, 1, 1)
    plt.plot(T, Q, label=leg)
    plt.legend()
    # plt.legend(labels=leg)
    plt.plot(T, [0 for t in T], color='k', linestyle="-")
    
    for t in interval_time:
        plt.vlines(ymax=max(q_list), ymin=min(q_list), x=t, color='k', linestyle="--")
    
    # plt.scatter(interval_time, q_list, marker=".")
    plt.xlabel("t",fontsize=fs)
    plt.ylabel("q",fontsize=fs,rotation=0)
    
    # T[-1]+0.5
    
    # plt.scatter([T[-1]+0.3], 0, marker=".")
    
    plt.subplot(4, 1, 2)
    plt.plot(T, np.array(V), label=leg)
    plt.legend()
    
    plt.xlabel("t",fontsize=fs)
    plt.ylabel("v",fontsize=fs,rotation=0)
    
    # plt.scatter([T[-1]+0.3], 0, marker=".")
    
    for t in interval_time:
        plt.vlines(ymax=max(V), ymin=-max(V), x=t, color='k', linestyle="--")
    plt.hlines(y = 0, xmin = 0, xmax = T[-1],color='k')
    # plt.plot(T, [self.vmax for t in T], color='k', linestyle="--")
    # plt.plot(T, [self.vmin for t in T], color='k', linestyle="--")
    # plt.text(0, 0.8*self.vmax, "vmax")
    # plt.text(0, 0.9*self.vmin, "vmin")

    plt.subplot(4, 1, 3)
    plt.plot(T, np.array(A), label=leg)
    plt.legend()
    
    
    plt.xlabel("t",fontsize=fs)
    plt.ylabel("a",fontsize=fs,rotation=0)
    for t in interval_time:
        plt.vlines(ymax=max(A), ymin=-max(A), x=t, color='k', linestyle="--")
        
    plt.hlines(y = 0, xmin = 0, xmax = T[-1],color='k')
    # plt.plot(T, [self.amax for t in T], color='k', linestyle="--")
    # plt.plot(T, [self.amin for t in T], color='k', linestyle="--")
    # plt.text(0, 0.8*self.amax, "amax")
    # plt.text(0, 0.9*self.amin, "amin")

    # plt.scatter([T[-1]+0.3], 0, marker=".")
    plt.subplot(4, 1, 4)
    # plt.plot(T, J)
    plt.plot(T, np.array(J), label=leg)
    plt.legend()
    
    
    for t in interval_time:
        plt.vlines(ymax=max(J), ymin=-max(J), x=t, color='k', linestyle="--")
    plt.xlabel("t",fontsize=fs)
    plt.ylabel("j",fontsize=fs,rotation=0)
    
    plt.hlines(y = 0, xmin = 0, xmax = T[-1],color='k')
    # plt.show()
    # plt.scatter([T[-1]+0.3], 0, marker=".")
    
    # plt.plot(T, [self.jmax for t in T], color='k', linestyle="--")
    # plt.plot(T, [self.jmin for t in T], color='k', linestyle="--")
    # plt.text(0, 0.8*self.jmax, "jmax")
    # plt.text(0, 0.9*self.jmin, "jmin")
        
def csv_traj(T,Q,V,A,J):
    jt_cols = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    df_ts = pd.DataFrame(np.array(T), columns=['ts'])
    df_pts = pd.DataFrame(np.array(Q).T, columns=['pts_'+c for c in jt_cols])
    df_vel = pd.DataFrame(np.array(V).T, columns=['vel_'+c for c in jt_cols])
    df_acc = pd.DataFrame(np.array(A).T, columns=['acc_'+c for c in jt_cols])
    df_jerk = pd.DataFrame(np.array(J).T, columns=['jerk_'+c for c in jt_cols])
    detailed_trajecory = pd.concat([df_ts, df_pts, df_vel, df_acc, df_jerk], axis=1)  
    detailed_trajecory.loc[:, "do_port"] = -1
    detailed_trajecory.loc[:, "do_state"] = 0
    return detailed_trajecory

if __name__ == "__main__":
    
    # thetas_list = np.array([
    #     [4.348656121726247e-05, 0.3795465492016311, -0.4681976595685353, 0.0005690269558297767, 0.07618667522241304, 0.00040767542555404657], 
    #     [-0.015800527788327207, -0.025090642380820836, -0.08890735778019697, -0.06657239207936708, 0.10120041963113187, 0.067174905005146], 


    #     [-0.16185417821349324, 0.03672665591082544, -0.20376520896312672, -1.1690283202830356, 0.6856897969646784, 1.1517834846287212], 
    #     [-0.32059013975131173, 0.05518567932774741, -0.4657308379429981, -0.6076988859902167, 0.42521877667443936, 1.2601100772479763],
    #     # [-0.31059013975131173, 0.05518567932774741, -0.4657308379429981, -0.6076988859902167, 0.42521877667443936, 1.2601100772479763],  
    #     [-0.16616128891045678, 0.07874467478115448, -0.4532413765450019, 0.06144333755615707, 0.3022420818835144, 0.5037576089470126], 
    #     [-0.4241962676401665, -0.3505023410626673, -0.03077934247233462, -1.4268037540738714, 1.15600768562094, 1.9281349945229973], 
    #     [-0.36949596178872945, -0.13278215130703017, -0.2632266731744879, -1.280658618205603, 0.909939334388764, 1.6174072261191503], 
    #     [0.22622596176027815, 0.11536005265140875, -0.20749251053186535, 1.239722122399956, 0.23953733926072684, -1.2269100555996726], 
    #     [0.4019229465756449, -0.5178934113759067, 0.5192924570545863, 0.5241773768142446, -1.270086748146829, -1.0461576781888733], 
    #     [0.6056871209565073, -0.7348597242067475, 0.47658443074376, -0.011704393724253033, -1.293859718301208, -0.8880375019864809], 
    #     [0.7408937416988454, -0.41981260890285715, -0.032605920317198, 0.0007418244311848676, -1.1144672710687191, -1.1100484244783004], 
    #     [0.8271388740219253, -0.016947901419748834, -0.5429190518317641, 0.0004026293869693124, -1.007006053662027, -1.483931712946494], 
    #     [1.0429570945833124, -0.40597113682434327, -0.9723790445806927, 0.5014945082206409, 1.4172149972591819, -1.6409691156927075], 
    #     [0.9392941433353849, -0.5936278279682863, -0.5323279864739633, 0.7985686033968079, 1.2573955538075121, -1.8588596174711536], 
    #     [0.8024948908427801, -0.811959009675492, -0.056045433434658745, 1.0922058421202487, 1.1964450467288756, -2.1557072828075587], 
    #     [0.6770230765338184, -1.0131904038364294, 0.3672310545421736, 1.311597907819594, 1.2280180274662493, -2.4282215075677263], 
    #     [0.550870907022334, -0.864750109931233, 0.6274054132386158, 0.004527369122760409, -1.3275822710194993, -0.7715123961173378]])
    c = getJoint()
    P0 = np.array([299.921,267.846,550,180,0,0])
    P1 = np.array([197.615742 , 142.844242 , 449.754  ,  -178.883649 , -19.633734  , -1.208145])
    P0_joint = c.transferCartesian2Joints(P0)
    P1_joint = c.transferCartesian2Joints(P1)
    print(P0_joint,P1_joint)
    thetas_list = np.array([[*P0_joint],
                            [*P1_joint]])
    curent_vel = np.array([0,0,0,0,0,0])
    target_vel = np.array([0,0,0,0,0,0])
    curent_acc = np.array([0,0,0,0,0,0])
    target_acc = np.array([0,0,0,0,0,0])
    T_s = 0.001
    q_boundary = np.array([[-2.94, -2.33, -1.2, -3.29, -1.98, -6.26], 
                           [ 2.94,  1.72,  3.47,  3.29,  1.98,  6.26]])
    v_boundary = np.array([[-5.71, -4.56, -5.71, -7.75, -6.97, -10.46],
                          [5.71, 4.56, 5.71, 7.75, 6.97, 10.46]])
    
    a_boundary = np.array([[-23*0.81, -19*0.82, -23*0.98, -31*0.99, -27*0.84, -39*0.9],
                          [23*0.81, 19*0.82, 23*0.98, 31*0.99, 27*0.84, 39*0.9]])
    j_boundary = np.array([[-230*0.81, -190*0.82, -230*0.98, -310*0.99, -270*0.84, -390*0.9],
                          [230*0.81, 190*0.82, 230*0.98, 310*0.99, 270*0.84, 390*0.9]])
    # a_boundary = np.array([-23, -19, -23, -31, -27, -39],
    #                       [23, 19, 23, 31, 27, 39])
    # j_boundary = np.array([-230, -190, -230, -310, -270, -390],
    #                       [230, 190, 230, 310, 270, 390])
    T, all_Q, all_V, all_A, all_J, duration,inter_time = multi_dimensional_p2p_ruckig(thetas_list[0],thetas_list[1],curent_vel,target_vel,curent_acc,target_acc,v_boundary,a_boundary,j_boundary,T_s)
    print("总时间:\n",duration)
    for idx in range(len(T)):
        print("第{}个点的关节角度:\n".format(idx+1), list(zip(*all_Q))[idx])
    # detailed_trajecory = csv_traj(T,all_Q,all_V,all_A,all_J)
    # output = detailed_trajecory.to_csv('traj.csv',index=False)
    c= AgilebotDriver()
    result = c.csv2executable_trajectory("/home/adt/RobotSystem/Sisyphe","traj")
    # for i in range(6):
    #     Q = all_Q[i]
    #     V = all_V[i]
    #     A = all_A[i]
    #     J = all_J[i]
    #     # T, Q, V, A, J, _, inter_time= multi_dimensional_ruckig(thetas_list[:,i:i+1], v_boundary[:,i:i+1], a_boundary[:,i:i+1], j_boundary[:,i:i+1], 0.001)
    
    #     draw_graph(T, Q, V, A, J, inter_time, thetas_list[:,i], "joint_"+str(i))
        
    #     plt.show()

    # q_boundary = np.array([[-2.9,  -2.5,    -4.8,    -3.3,    -2.1,   -6.2], 
    #                        [2.9,  2.5,    4.8,    3.3,    2.1,   6.2]])
                               
    # v_boundary = np.array([[-5.16, -4.32,   -5.72,   -7.6,    -7.6,    -13.96],
    #                        [5.16, 4.32,   5.72,   7.6,    7.6,    13.96]])
    
    # a_boundary = np.array([[-21.52,-18.03,  -23.85,  -31.99,  -31.7,   -58.17],
    #                        [21.52,18.03,  23.85,  31.99,  31.7,   58.17]])
    
    # j_boundary = np.array([[-179,  -150,    -198,    -266,    -264,    -484],
    #                        [179,  150,    198,    266,    264,    484]])
    
    # a = thetas_list[:,0:1]
    
    # T, all_Q, all_V, all_A, all_J, _, inter_time= multi_dimensional_ruckig(thetas_list, v_boundary, a_boundary, j_boundary, 0.001)
    
    # diff = np.array(T[1:]) - np.array(T[:-1])
    # for i in range(6):
    #     Q = all_Q[i]
    #     V = all_V[i]
    #     A = all_A[i]
    #     J = all_J[i]
    #     # T, Q, V, A, J, _, inter_time= multi_dimensional_ruckig(thetas_list[:,i:i+1], v_boundary[:,i:i+1], a_boundary[:,i:i+1], j_boundary[:,i:i+1], 0.001)
    
    #     draw_graph(T, Q, V, A, J, inter_time, thetas_list[:,i], "joint_"+str(i))
        
    #     plt.show()
    # # # diff = np.array(T[1:]) - np.array(T[:-1])
    # for i in range(6):
    #     # i=1
    #     # Q = all_Q[i]
    #     # V = all_V[i]
    #     # A = all_A[i]
    #     # J = all_J[i]
        
    #     print(thetas_list[:,i:i+1])
    #     T, Q, V, A, J, _, inter_time= multi_dimensional_ruckig(thetas_list[:,i:i+1], v_boundary[:,i:i+1], a_boundary[:,i:i+1], j_boundary[:,i:i+1], 0.001)
    
    #     draw_graph(T, Q[0], V[0], A[0], J[0], inter_time, thetas_list[:,i], "joint_"+str(i))
        
    #     plt.show()