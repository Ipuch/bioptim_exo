#from main import main as my_kcc
import enum

import main
import numpy as np
from data.enums import TasksKinova
import matplotlib.pyplot as plt
import time
import biorbd
from models import enums

from ezc3d import c3d


def sensibility_param_id(nb_frame_param_step, end_loop, step, nbr_colum,use_analytical_jacobians):


    #number of lines
    nbr_lines=int(((((end_loop-1)-nb_frame_param_step)//step)//nbr_colum)+1)
    plt.figure(1)
    index=1

    for i in range (nb_frame_param_step,end_loop,step):

        # get RMS
        RMS = main.main(TasksKinova.DRINK, False, False, i, use_analytical_jacobians)[2].values()
        # [[RMS_x],[RMS_y],[RMS_z],[RMS_tot], max_marker, gain_time]

        nb_frames=len(RMS.mapping["rmse_x"])
        time_IK = RMS.mapping["gain_time"]

        # divide the graphic
        plt.subplot(nbr_lines,nbr_colum,index)
        # plot RMS curve for each frame
        plt.grid(True)
        plt.title("boire , nbre_frame_param_step vaut %r" %i)
        plt.plot([p for p in range(nb_frames)], RMS.mapping["rmse_x"], "b", label="RMS_x")
        plt.plot([p for p in range(nb_frames)], RMS.mapping["rmse_y"], "y", label="RMS_y")
        plt.plot([p for p in range(nb_frames)], RMS.mapping["rmse_z"], "g", label="RMS_z")
        plt.plot([p for p in range(nb_frames)], RMS.mapping["rmse_tot"], "r", label="RMS_tot")
        plt.xlabel('Frame')
        plt.ylabel('Valeurs (m)')
        plt.legend()

        print("max_marker = ", RMS.mapping["max_marker"])
        print("-----LOOP-----" , index, "-----FINISHED-------Use_analytical_jacobian-----", use_analytical_jacobians,"-----")

        # plt.figure(2)
        # plt.hist( RMS.mapping["gain_time"], [i for i in range(len(RMS.mapping["gain_time"]))])
        # plt.xlabel("number of the IK")
        # plt.ylabel("gain time in second")
        # plt.legend()

        index+=1

    plt.show()

    return time_IK

def show_RMS(task : TasksKinova,
             show_animation: bool,
             export_model: bool,
             nb_frame_param_step: int,
             use_analytical_jacobians: bool,

):
    plt.figure(1)
    # get RMS
    RMS = main.main(task, show_animation, export_model, nb_frame_param_step, use_analytical_jacobians)[2].values()
    # [[RMS_x],[RMS_y],[RMS_z],[RMS_tot], max_marker, gain_time]
    nb_frames = len(RMS.mapping["rmse_x"])

    plt.grid(True)
    plt.title("boire , nbre_frame_param_step vaut %r" %nb_frame_param_step)
    plt.plot([p for p in range(nb_frames)], RMS.mapping["rmse_x"], "b", label="RMS_x")
    plt.plot([p for p in range(nb_frames)], RMS.mapping["rmse_y"], "y", label="RMS_y")
    plt.plot([p for p in range(nb_frames)], RMS.mapping["rmse_z"], "g", label="RMS_z")
    plt.plot([p for p in range(nb_frames)], RMS.mapping["rmse_tot"], "r", label="RMS_tot")
    plt.xlabel('Frame')
    plt.ylabel('Valeurs (m)')
    plt.legend()

    print("max_marker = ", RMS.mapping["max_marker"])

    plt.show()

def compare_RMS(task : TasksKinova,
                show_animation: bool,
                export_model: bool,
                nb_frame_param_step: int,
):
    plt.figure(1)
    L=[True,False]
    index=1

    for i in L:
        # get RMS
        RMS = main.main(task, show_animation, export_model, nb_frame_param_step, i)[2].values()
        # [[RMS_x],[RMS_y],[RMS_z],[RMS_tot], max_marker, gain_time]
        nb_frames = len(RMS.mapping["rmse_x"])

        plt.subplot(1,2,index)
        plt.grid(True)
        plt.title("boire ,  use analytical jacobian %r" %i )
        plt.plot([p for p in range(nb_frames)], RMS.mapping["rmse_x"], "b", label="RMS_x")
        plt.plot([p for p in range(nb_frames)], RMS.mapping["rmse_y"], "y", label="RMS_y")
        plt.plot([p for p in range(nb_frames)], RMS.mapping["rmse_z"], "g", label="RMS_z")
        plt.plot([p for p in range(nb_frames)], RMS.mapping["rmse_tot"], "r", label="RMS_tot")
        plt.xlabel('Frame')
        plt.ylabel('Valeurs (m)')
        plt.legend()
        index+=1

    plt.show()


def time_spend_entire_script (nb_frame_param_step, end_loop, step, nbr_colum,use_analytical_jacobians):
    start = time.time()
    time_IK = sensibility_param_id(nb_frame_param_step,end_loop,step,nbr_colum,use_analytical_jacobians)
    end = time.time()
    print("running time = ", end-start, "seconds")
    return (end-start,time_IK)

def time_spend_entire_script_comparison(nb_frame_param_step, end_loop, step, nbr_colum):
    L=[True,False]
    t=[]
    time_list=[]
    for i in L:
        time_entire,gain = time_spend_entire_script(nb_frame_param_step,end_loop,step,nbr_colum,i)
        t.append(time_entire)
        time_list.append(gain)
    time_difference=t[1]-t[0]
    print("Gain time on the entire script = ",time_difference, "seconds")

    return time_difference

def show_q(task,show_animation,export_model, nb_frame_param_step):
    #get the model
    biorbd_model = biorbd.Model(enums.Models.WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_6_DOF_SUPPORT_VARIABLES.value )
    model_dofs = [dof.to_string() for dof in biorbd_model.nameDof()]

    # execute main twice with and without an analytic Jacobian
    L = [True, False]
    q_list = []
    for i in L:
        #pos shape is [22x710]
        pos= main.main( task, show_animation, export_model, nb_frame_param_step,i)[0]
        q_list.append( pos )

    nb_frame = np.shape(q_list[0])[1]
    diff_q = q_list[0] - q_list[1]

    index = 1
    for i in range(np.shape(diff_q)[0]):
            plt.figure(i)
            plt.subplot(4, 4, index)
            plt.plot([k for k in range(nb_frame)], q_list[0][i,:],"b", label="q_analytic_%r" %i )
            plt.plot([k for k in range(nb_frame)], q_list[1][i,:], "r", label="q_numeric_%r" %i  )
            plt.xlabel('Frame')
            plt.ylabel(' q ')
            plt.legend()
            plt.grid(True)
            index+=1

    plt.show()

def show_numeric_jacobian(task, nb_frame_param_step):
    L=[True,False]
    for j in L:
        if j == True:
            kcc = main.prepare_kcc(task=task,nb_frame_param_step= nb_frame_param_step, use_analytical_jacobians= j)[2]
            jacobians_used_analytic, q_out = kcc.solve(threshold=1e-5)[2], kcc.solve(threshold=1e-5)[0]

            #list which contains each matrix of step2 used in the solve method, often len(list) is 3
        else:
            kcc = main.prepare_kcc(task=task, nb_frame_param_step=nb_frame_param_step, use_analytical_jacobians=j)[2]
            jacobians_used_numeric = kcc.solve(threshold=1e-5)[2]

    #np.delete(q_out,[10,11,12,13,14,15])
    # for l in jacobians_used_analytic:
    #     l= l * q_out

    nb_line = np.shape(jacobians_used_numeric[0])[0]
    nb_column = np.shape(jacobians_used_numeric[0])[1]
    for i in range(len(jacobians_used_numeric)):
        index1= 1
        for k in range(nb_column):
            plt.figure(i)
            plt.subplot(4,4,index1)
            plt.scatter([p for p in range(nb_line)], jacobians_used_numeric[i][:, k])
            plt.scatter([p for p in range(nb_line)], jacobians_used_analytic[i][:, k])
            index1 += 1



    plt.show()




    #print(jacobians_used)







if __name__ == "__main__":
    show_q(task= TasksKinova.DRINK ,show_animation=False,export_model=False, nb_frame_param_step= 100)