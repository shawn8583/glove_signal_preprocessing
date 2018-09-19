import csv
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

filename_1 = '/home/shawn/ros_ws/src/robot_skin/icra_glove/data/training_sets_200s_whole_csv/1_luo/1.csv'

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

x_train = np.array([taxel0,taxel1,taxel2,taxel3,taxel4,taxel5,taxel6,taxel7,taxel8,taxel9,taxel10,taxel11])

# plot data
plt.figure('Draw')
plt.title('gesture_1_csv_plot')
plt.xlabel('time sequence')
plt.ylabel('taxel value')
plt.plot(sequence, x_train[0], label='taxel_1')
# plt.plot(sequence, x_train[1], label='taxel_2')
# plt.plot(sequence, x_train[2], label='taxel_3')
# plt.plot(sequence, x_train[3], label='taxel_4')
# plt.plot(sequence, x_train[4], label='taxel_5')
# plt.plot(sequence, x_train[5], label='taxel_6')
# plt.plot(sequence, x_train[6], label='taxel_7')
# plt.plot(sequence, x_train[7], label='taxel_8')
# plt.plot(sequence, x_train[8], label='taxel_9')
# plt.plot(sequence, x_train[9], label='taxel_10')
# plt.plot(sequence, x_train[10], label='taxel_11')
# plt.plot(sequence, x_train[11], label='taxel_12')
plt.show()
plt.legend()

# Sampling and Smoothing
with open(filename_1) as f1:
    raw_data = csv.reader(f1)

    # Sampling 1/10th of the Data
    one_tenth = raw_data.sample(frac = .1, random_state=np.random.randint(10))

    #reordering by time sequence
    one_tenth.index.name = None
    one_tenth = one_tenth.sort_values(by=['field.header.seq'], ascending=[True])
    axes = one_tenth.plot('field.header.seq', 'field.data0', legend = False, title = 'Sampled PLot')
    axes.legend = None
    axes.set_ylabel('Taxel Value')
