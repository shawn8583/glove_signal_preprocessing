import csv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

filename_1 = '/home/shawn/ros_ws/src/robot_skin/icra_glove/data/training_sets_200s_whole_csv/1_luo/1.csv'

# very simple plotting of the original data
raw_data = pd.read_csv('/home/shawn/ros_ws/src/robot_skin/icra_glove/data/code_test_data.csv', sep=',', names=['data0','data1','data2','data3','data4','data5','data6','data7','data8','data9','data10','data11'])
raw_data['sequence'] = range(0, len(raw_data.index.values))
print raw_data.head()
fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax1.set_xlabel('time sequence')
ax1.set_ylabel('taxel[0]_value')
ax1.set_title('gesture_1 taxel0')
ax1.plot(raw_data.sequence, raw_data.data0)
plt.show()

# Sampling 1/10th of the Data
one_tenth = raw_data.sample(frac = .1, random_state=np.random.randint(10))
print one_tenth.head()

# Reordering by time sequence and plot sampled data
one_tenth.index.name = None
one_tenth = one_tenth.sort_values(by=['sequence'], ascending=[True])
axes = one_tenth.plot('sequence', 'data0', legend = False, title = 'Sampled Plot')
axes.legend = None
axes.set_ylabel('Taxel Value')
plt.show()

# Plotting Original Data vs Sampled Data by Subplot
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))
axes[0].plot(raw_data.sequence, raw_data.data0)
axes[0].set_title('Original Plot')
axes[1].plot(one_tenth.sequence, one_tenth.data0)
axes[1].set_title('Sampled Plot')
plt.show()

# **Implement Rolling Mean and **Plot Original Data vs Sampled vs Rolling Mean (Subplot)
raw_data['Rolling_Mean'] = raw_data['data0'].rolling(window = 80).mean()
fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize  = (15, 5))
axes[0].plot(raw_data.sequence, raw_data.data0)
axes[0].set_title('Original')
axes[1].plot(one_tenth.sequence, one_tenth.data0)
axes[1].set_title('Sampled')
axes[2].plot(raw_data.sequence, raw_data.Rolling_Mean)
axes[2].set_title('Smoothed (Rolling_Mean)')
plt.show()

# Plotting Original Data and Smoothed Data on Same Plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(raw_data.sequence, raw_data.Rolling_Mean, color=(0,0,0), linewidth = 4, alpha = .9, label='Smoothed')
ax.plot(raw_data.sequence, raw_data.data0, color=(1,0,0), label='Original')
ax.set_title('Original and Smoothed data')
ax.set_xlabel('Time Sequence')
ax.set_ylabel('Taxel0 data')
ax.legend(loc='lower right')
plt.show()



# ----------------------------------------------------------------------------------------------------------
# Another way to read and plot csv files using "import csv", but it's way more complicated than pandas
# -----------------------------------------------------------------------------------------------------------
#
# with open(filename_1) as f1:
#     reader = csv.reader(f1)
#     header_row = next(reader)
#     seq = 0
#     sequence, taxel0, taxel1, taxel2, taxel3, taxel4, taxel5, taxel6, taxel7, taxel8, taxel9, taxel10, taxel11 = [],[],[],[],[],[],[],[],[],[],[],[],[]
#     maxi = 1024
#     for row in reader:
#         sequence.append(seq)
#         seq += 1
#         data0 = float(row[4])
#         taxel0.append(data0)
#         data1 = float(row[5])
#         taxel1.append(data1)
#         data2 = float(row[6])
#         taxel2.append(data2)
#         data3 = float(row[7])
#         taxel3.append(data3)
#         data4 = float(row[8])
#         taxel4.append(data4)
#         data5 = float(row[9])
#         taxel5.append(data5)
#         data6 = float(row[10])
#         taxel6.append(data6)
#         data7 = float(row[11])
#         taxel7.append(data7)
#         data8 = float(row[12])
#         taxel8.append(data8)
#         data9 = float(row[13])
#         taxel9.append(data9)
#         data10 = float(row[14])
#         taxel10.append(data10)
#         data11 = float(row[15])
#         taxel11.append(data11)

# x_train = np.array([taxel0,taxel1,taxel2,taxel3,taxel4,taxel5,taxel6,taxel7,taxel8,taxel9,taxel10,taxel11])

# # plot data
# plt.figure('Draw')
# plt.title('gesture_1_csv_plot')
# plt.xlabel('time sequence')
# plt.ylabel('taxel value')
# plt.plot(sequence, x_train[0], label='taxel_1')
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
# plt.show()
# plt.legend()
