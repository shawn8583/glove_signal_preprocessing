import csv
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn import preprocessing
import numpy as np
import scipy.signal as signal

filename_1 = '/home/shawn/ros_ws/src/robot_skin/icra_glove/data/training_sets_200s_whole_csv/1_luo/1.csv'
#filename_2 = '2.csv'
#filename_3 = '3.csv'
#filename_4 = '4.csv'
#filename_5 = '5.csv'
#filename_6 = '6.csv'
#filename_7 = '7.csv'
#filename_8 = '8.csv'
#filename_9 = '9.csv'
#filename_N = 'N.csv'
#filename_C = 'C.csv'
#filename_k = 'K.csv'

with open(filename_1) as f1:
    reader = csv.reader(f1)
    header_row = next(reader)
    seq = 0
    sequence, taxel0, taxel1, taxel2, taxel3, taxel4, taxel5, taxel6, taxel7, taxel8, taxel9, taxel10, taxel11 = [],[],[],[],[],[],[],[],[],[],[],[],[]
#    maxi = 1024
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

# filter signal.medfilt()
data_raw = np.array([taxel0,taxel1,taxel2,taxel3,taxel4,taxel5,taxel6,taxel7,taxel8,taxel9,taxel10,taxel11])
# filtered = signal.medfilt(data_raw, 11)
filtered = 

# # filter 

# # plot preprocessed data
# plt.figure('Draw')
# plt.title('gesture_1_csv_plot')
# plt.plot(sequence, filtered[0], label='taxel_1', color='r')
# plt.plot(sequence, filtered[1], label='taxel_2', color='k')
# plt.plot(sequence, filtered[2], label='taxel_3', color='yellow')
# plt.plot(sequence, filtered[3], label='taxel_4', color='green')
# plt.plot(sequence, filtered[4], label='taxel_5', color='cyan')
# plt.plot(sequence, filtered[5], label='taxel_6', color='b')
# plt.plot(sequence, filtered[6], label='taxel_7', color='m')
# plt.plot(sequence, filtered[7], label='taxel_8', color='cadetblue')
# plt.plot(sequence, filtered[8], label='taxel_9', color='orange')
# plt.plot(sequence, filtered[9], label='taxel_10', color='chocolate')
# plt.plot(sequence, filtered[10], label='taxel_11', color='deeppink')
# plt.plot(sequence, filtered[11], label='taxel_12'
         , color='indigo')
plt.legend()

plt.xlabel('time sequence')
plt.ylabel('taxel value')
plt.show()
