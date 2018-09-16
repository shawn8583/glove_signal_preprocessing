import csv
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl

# plot gesture 1 of all 5 participants in one diagram.
participant_1 = '/home/shawn/ros_ws/src/robot_skin/icra_glove/data/training_sets_200s_whole_csv/1_luo/1.csv'
participant_2 = '/home/shawn/ros_ws/src/robot_skin/icra_glove/data/training_sets_200s_whole_csv/2_shawn/1.csv'
participant_3 = '/home/shawn/ros_ws/src/robot_skin/icra_glove/data/training_sets_200s_whole_csv/3_luo2/1.csv'
participant_4 = '/home/shawn/ros_ws/src/robot_skin/icra_glove/data/training_sets_200s_whole_csv/4_wang/1.csv'
participant_5 = '/home/shawn/ros_ws/src/robot_skin/icra_glove/data/training_sets_200s_whole_csv/5_yaxiang/1.csv'

def plot_participant(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        seq = 1
        sequence, taxel0, taxel1, taxel2, taxel3, taxel4, taxel5, taxel6, taxel7, taxel8, taxel9, taxel10, taxel11 = [],[],[],[],[],[],[],[],[],[],[],[],[]
        for row in reader:
            sequence.append(seq)
            seq += 1
            data0 = float(row[4])
            taxel0.append(data0)
            data1 = float(row[5])
            taxel1.append(data1)
            data2 = float(row[6])
            taxel2.append(data2)
            data3 = float(row[7])
            taxel3.append(data3)
            data4 = float(row[8])
            taxel4.append(data4)
            data5 = float(row[9])
            taxel5.append(data5)
            data6 = float(row[10])
            taxel6.append(data6)
            data7 = float(row[11])
            taxel7.append(data7)
            data8 = float(row[12])
            taxel8.append(data8)
            data9 = float(row[13])
            taxel9.append(data9)
            data10 = float(row[14])
            taxel10.append(data10)
            data11 = float(row[15])
            taxel11.append(data11)

    plt.plot(sequence, taxel0, label='taxel_1', color='r')
    plt.plot(sequence, taxel1, label='taxel_2', color='k')
    plt.plot(sequence, taxel2, label='taxel_3', color='yellow')
    plt.plot(sequence, taxel3, label='taxel_4', color='green')
    plt.plot(sequence, taxel4, label='taxel_5', color='cyan')
    plt.plot(sequence, taxel5, label='taxel_6', color='b')
    plt.plot(sequence, taxel6, label='taxel_7', color='m')
    plt.plot(sequence, taxel7, label='taxel_8', color='cadetblue')
    plt.plot(sequence, taxel8, label='taxel_9', color='orange')
    plt.plot(sequence, taxel9, label='taxel_10', color='chocolate')
    plt.plot(sequence, taxel10, label='taxel_11', color='deeppink')
    plt.plot(sequence, taxel11, label='taxel_12', color='indigo')

if __name__=='__main__':
    plt.figure('Draw')
    plt.title('gesture_1_csv_plot')
    plot_participant(participant_1)
    plt.xlabel('time sequence')
    plt.ylabel('taxel value')
    plt.legend()
    plt.show()
    plot_participant(participant_3)
    plt.xlabel('time sequence')
    plt.ylabel('taxel value')
    plt.legend()
    plt.show()

