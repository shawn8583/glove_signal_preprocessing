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

# Reference and Tutorial about the Sampling and Smoothing methods this script implemented:
# https://github.com/mGalarnyk/Python_Tutorials/blob/master/Time_Series/Part1_Time_Series_Data_BasicPlotting.ipynb

# Very simple plotting of the original data
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
raw_data['Rolling_Mean0'] = raw_data['data0'].rolling(window = 40).mean()   
raw_data['Rolling_Mean1'] = raw_data['data1'].rolling(window = 40).mean() 
raw_data['Rolling_Mean2'] = raw_data['data2'].rolling(window = 40).mean() 
raw_data['Rolling_Mean3'] = raw_data['data3'].rolling(window = 40).mean() 
raw_data['Rolling_Mean4'] = raw_data['data4'].rolling(window = 40).mean() 
raw_data['Rolling_Mean5'] = raw_data['data5'].rolling(window = 40).mean() 
raw_data['Rolling_Mean6'] = raw_data['data6'].rolling(window = 40).mean() 
raw_data['Rolling_Mean7'] = raw_data['data7'].rolling(window = 40).mean() 
raw_data['Rolling_Mean8'] = raw_data['data8'].rolling(window = 40).mean() 
raw_data['Rolling_Mean9'] = raw_data['data9'].rolling(window = 40).mean() 
raw_data['Rolling_Mean10'] = raw_data['data10'].rolling(window = 40).mean() 
raw_data['Rolling_Mean11'] = raw_data['data11'].rolling(window = 40).mean() 

raw_data_without_fist_100_rows = raw_data.drop(raw_data.index[0,100,1])
print raw_data.head()
print raw_data.tail()
print raw_data_without_fist_100_rows.head()

fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize  = (15, 5))
axes[0].plot(raw_data.sequence, raw_data.data0)
axes[0].set_title('Original')
axes[1].plot(one_tenth.sequence, one_tenth.data0)
axes[1].set_title('Sampled')
axes[2].plot(raw_data.sequence, raw_data.Rolling_Mean0)
axes[2].set_title('Smoothed (Rolling_Mean)')
plt.show()

# Plot all 12 taxel data after implemented Sampling and Rolling Mean Smoothing Method
x_train = np.array([raw_data.Rolling_Mean0, raw_data.Rolling_Mean1, raw_data.Rolling_Mean2, raw_data.Rolling_Mean3, raw_data.Rolling_Mean4, raw_data.Rolling_Mean5, raw_data.Rolling_Mean6, raw_data.Rolling_Mean7, raw_data.Rolling_Mean8, raw_data.Rolling_Mean9, raw_data.Rolling_Mean10, raw_data.Rolling_Mean11])
x_scaled = preprocessing.scale(x_train)

fig_all_12 = plt.figure(2)
ax1_12 = fig_all_12.add_subplot(111)
ax1_12.set_xlabel('time sequence')
ax1_12.set_ylabel('taxel values')
ax1_12.set_title('gesture_1 with all 12 signals')
ax1_12.plot(raw_data.sequence, x_train[0], label='taxel0')
ax1_12.plot(raw_data.sequence, x_train[1], label='taxel1')
ax1_12.plot(raw_data.sequence, x_train[2], label='taxel2')
ax1_12.plot(raw_data.sequence, x_train[3], label='taxel3')
ax1_12.plot(raw_data.sequence, x_train[4], label='taxel4')
ax1_12.plot(raw_data.sequence, x_train[5], label='taxel5')
ax1_12.plot(raw_data.sequence, x_train[6], label='taxel6')
ax1_12.plot(raw_data.sequence, x_train[7], label='taxel7')
ax1_12.plot(raw_data.sequence, x_train[8], label='taxel8')
ax1_12.plot(raw_data.sequence, x_train[9], label='taxel9')
ax1_12.plot(raw_data.sequence, x_train[10], label='taxel10')
ax1_12.plot(raw_data.sequence, x_train[11], label='taxel11')
plt.show()

# Plotting Original Data and Smoothed Data on Same Plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(raw_data.sequence, raw_data.Rolling_Mean0, color=(0,0,0), linewidth = 4, alpha = .9, label='Smoothed')
ax.plot(raw_data.sequence, raw_data.data0, color=(1,0,0), label='Original')
ax.set_title('Original and Smoothed data')
ax.set_xlabel('Time Sequence')
ax.set_ylabel('Taxel0 data')
ax.legend(loc='lower right')
plt.show()


# Linear Regression
# Keeps having problem saying that "Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."
# raw_data = raw_data[(raw_data['sequence'] >= 800) & (raw_data['sequence'] <= 1200)]
raw_data = raw_data.sort_values(by=['sequence'], ascending=[True])
raw_data.head()
model = LinearRegression().fit(raw_data[['sequence']], raw_data[['Rolling_Mean0']])
m = model.coef_[0]
b = model.intercept_
print 'y = ', round(m[0], 2), 'x +', round(b[0], 2)
predictions = model.predict(raw_data.data0)
predictions[0:5]
predictions = pd.DataFrame(data = predictions, index = raw_data.index.values, columns = ['Pred'])
print predictions.head()
