import csv
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl

# plot gesture 1 of all 5 participants in one diagram.
participant_1 = '/home/shawn/ros_ws/src/robot_skin/icra_glove/data/training_sets_200s_whole_csv/1_luo/1.csv'
participant_2 = '/home/shawn/ros_ws/src/robot_skin/icra_glove/data/training_sets_200s_whole_csv/2_shawn/1.csv'
participant_3 = '/home/shawn/ros_ws/src/robot_skin/icra_glove/data/training_sets_200s_whole_csv/3_luo2/1.csv'
participant_4 = '/home/shawn/ros_ws/src/robot_skin/icra_glove/data/training_sets_200s_whole_csv/4_wang/1.csv'
participant_5 = '/home/shawn/ros_ws/src/robot_skin/icra_glove/data/training_sets_200s_whole_csv/5_yaxiang.csv'

# load participant 1
with open(participant_1) as f1:
    reader = csv.reader(f1)
    header_row = next(reader)
    seq1 = 1
    sequence1, taxel1_0, taxel1_1, taxel1_2, taxel1_3, taxel1_4, taxel1_5, taxel1_6, taxel1_7, taxel1_8, taxel1_9, taxel1_10, taxel1_11 = [],[],[],[],[],[],[],[],[],[],[],[],[]
    maxi = 1 # didn't do normalization
    for row in reader:
        sequence1.append(seq1/maxi)
        seq1 += 1
        data1_0 = float(row[4])
        taxel1_0.append(data1_0/maxi)
        data1_1 = float(row[5])
        taxel1_1.append(data1_1/maxi)
        data1_2 = float(row[6])
        taxel1_2.append(data1_2/maxi)
        data1_3 = float(row[7])
        taxel1_3.append(data1_3/maxi)
        data1_4 = float(row[8])
        taxel1_4.append(data1_4/maxi)
        data1_5 = float(row[9])
        taxel1_5.append(data1_5/maxi)
        data1_6 = float(row[10])
        taxel1_6.append(data1_6/maxi)
        data1_7 = float(row[11])
        taxel1_7.append(data1_7/maxi)
        data1_8 = float(row[12])
        taxel1_8.append(data1_8/maxi)
        data1_9 = float(row[13])
        taxel1_9.append(data1_9/maxi)
        data1_10 = float(row[14])
        taxel1_10.append(data1_10/maxi)
        data1_11 = float(row[15])
        taxel1_11.append(data1_11/maxi)

# load participant 2
with open(participant_2) as f2:
    reader = csv.reader(f2)
    header_row = next(reader)
    seq2 = 1
    sequence2, taxel2_0, taxel2_1, taxel2_2, taxel2_3, taxel2_4, taxel2_5, taxel2_6, taxel2_7, taxel2_8, taxel2_9, taxel2_10, taxel2_11 = [],[],[],[],[],[],[],[],[],[],[],[],[]
    maxi = 1 # didn't do normalization
    for row in reader:
        sequence2.append(seq2/maxi)
        seq2 += 1 
        data2_0 = float(row[4])
        taxel2_0.append(data2_0/maxi)
        data2_1 = float(row[5])
        taxel2_1.append(data2_1/maxi)
        data2_2 = float(row[6])
        taxel2_2.append(data2_2/maxi)
        data2_3 = float(row[7])
        taxel2_3.append(data2_3/maxi)
        data2_4 = float(row[8])
        taxel2_4.append(data2_4/maxi)
        data2_5 = float(row[9])
        taxel2_5.append(data2_5/maxi)
        data2_6 = float(row[10])
        taxel2_6.append(data2_6/maxi)
        data2_7 = float(row[11])
        taxel2_7.append(data2_7/maxi)
        data2_8 = float(row[12])
        taxel2_8.append(data2_8/maxi)
        data2_9 = float(row[13])
        taxel2_9.append(data2_9/maxi)
        data2_10 = float(row[14])
        taxel2_10.append(data2_10/maxi)
        data2_11 = float(row[15])
        taxel2_11.append(data2_11/maxi)


# plot data
plt.figure('Draw')
plt.title('gesture_1_csv_plot')
plt.plot(sequence1, taxel1_0, label='taxel_1', color='r')
plt.plot(sequence1, taxel1_1, label='taxel_2', color='k')
plt.plot(sequence1, taxel1_2, label='taxel_3', color='yellow')
plt.plot(sequence1, taxel1_3, label='taxel_4', color='green')
plt.plot(sequence1, taxel1_4, label='taxel_5', color='cyan')
plt.plot(sequence1, taxel1_5, label='taxel_6', color='b')
plt.plot(sequence1, taxel1_6, label='taxel_7', color='m')
# plt.plot(sequence1, taxel1_7, label='taxel_8')
# plt.plot(sequence1, taxel1_8, label='taxel_9')
# plt.plot(sequence1, taxel1_9, label='taxel_10')
# plt.plot(sequence1, taxel1_10, label='taxel_11')
# plt.plot(sequence1, taxel1_11, label='taxel_12')

plt.plot(sequence2, taxel2_0, color='r')
plt.plot(sequence2, taxel2_1, color='k')
plt.plot(sequence2, taxel2_2, color='yellow')
plt.plot(sequence2, taxel2_3, color='green')
plt.plot(sequence2, taxel2_4, color='cyan')
plt.plot(sequence2, taxel2_5, color='b')
plt.plot(sequence2, taxel2_6, color='m')
# plt.plot(sequence2, taxel2_7)
# plt.plot(sequence2, taxel2_8)
# plt.plot(sequence2, taxel2_9)
# plt.plot(sequence2, taxel2_10)
# plt.plot(sequence2, taxel2_11)


plt.legend()
plt.xlabel('time sequence')
plt.ylabel('taxel value')
plt.show()
