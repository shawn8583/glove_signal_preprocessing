import os,glob
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import ipdb

def run():
    hz  = 170
    total_sec = 200
    split_sec = 10
    gap = 1
    gesture_path = '/home/birl_wu/Desktop/glove/'

    exp_dirs = [i for i in glob.glob(os.path.join(gesture_path,'*')) if os.path.isdir(i)]
    exp_dirs.sort()
    for exp_dir in exp_dirs:
        print(exp_dir)
        csvs = glob.glob(os.path.join(
            exp_dir,
            'sample_*',
            'sample_*.csv',
        ))
        csvs.sort()
        for csv in csvs:
            print csv
            csv_path = os.path.dirname(csv)
            path_postfix = os.path.relpath(csv_path, gesture_path)
            output_dir = os.path.join(gesture_path,'splitted_file_gap_%s'%gap,path_postfix)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            df = pd.read_csv(csv, sep=',')
            iplot = int(total_sec/split_sec)
            fig, axarr = plt.subplots(nrows=iplot, ncols=1, sharex=True,figsize=(16,3*iplot))
            for i in range(int(total_sec/split_sec)):
                if os.path.exists(os.path.join(output_dir,'%s.csv'%(i+1))):
                    print('file exist, skip')
                    continue                
                try:
                    st = i*split_sec*hz
                    en = (i+1)*split_sec*hz
                    temp_df = df.iloc[st:en, :]
                    temp_df_thin = temp_df.iloc[::gap] # add more powerful feature engineering here!!

                    val = temp_df_thin.values
                    min_max_scaler = preprocessing.MinMaxScaler()
                    val_scaled = min_max_scaler.fit_transform(val)
                    temp_df_thin = pd.DataFrame(val_scaled, columns=temp_df_thin.columns)
                    temp_df_thin.to_csv(os.path.join(output_dir,'%s.csv'%(i+1)), index=False)
                    temp_df_thin[temp_df_thin.columns[1:]].plot(ax=axarr[i], yticks= [], legend=False)
                except:
                    print ("something wrong here, please check!")
                    ipdb.set_trace()
            plt.savefig(os.path.join(output_dir, 'samples.pdf'),dpi=300)
            ipdb.set_trace()            
if __name__=='__main__':
    run()
