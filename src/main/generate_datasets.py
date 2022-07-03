# %%
import numpy as np 
import pandas as pd
import glob as gb
from tqdm import tqdm
DATA_PATH = 'D:/OpenData/Kaggle/smartphone-decimeter-2022'

trains = gb.glob(f"{DATA_PATH}/bronze/train/*/*/*")
tests = gb.glob(f"{DATA_PATH}/bronze/test/*/*/*")

places = [
    'LAX-1', 'LAX-2', 'LAX-3', 'LAX-5', 'MTV-1', 'MTV-2', 'MTV-3', 'OAK-1', 'OAK-2'
 'SFO-1', 'SFO-2', 'SJC-1', 'SJC-2', 'SVL-1', 'SVL-2'
]
# %%
# pathの順序を走行場所ごとになるように変更
def make_paths_each_place(paths):
    new_paths = []
    for place in places:
        tmp_paths = []
        for path in paths:
            prev_place_path = path.split('US-')[0] # 走行場所より前のpath
            place_name = path.split('US-')[-1].split('\\')[0] # 走行場所抽出
            next_place_path = path.split('\\')[2:][0] # 走行場所より後のpath
            if place == place_name: 
                tmp_paths.append(prev_place_path+'US-'+place_name+'/'+next_place_path+'/')
        new_paths.append(tmp_paths)
    return new_paths

# %%
tr_paths = [np.unique(np.array(places_path)) for places_path in make_paths_each_place(trains)]
te_paths = [np.unique(np.array(places_path)) for places_path in make_paths_each_place(tests)]
tr_paths[0:5]
# %%
# 結合の識別元
def naming_collection_and_phone(df, path):
    df['collectionName'] = path.split('\\')[-1].split('/')[0]
    df['phoneName'] = path.split('/')[-2]
    return df

def generate_datasets(paths, dir_name):
    gnss_list, gt_list = [], []
    mag_list, acc_list, gyro_list = [], [], []
    for phone_paths in tqdm(paths):
        for path in phone_paths:
            gnss_df = pd.read_csv(f'{path}/device_gnss.csv')
            gnss_df = naming_collection_and_phone(gnss_df, path)
            gnss_list.append(gnss_df)

            imu_df = pd.read_csv(f'{path}/device_imu.csv')
            imu_df = naming_collection_and_phone(imu_df, path)
            mag_df = imu_df[imu_df['MessageType']=='UncalMag']
            acc_df = imu_df[imu_df['MessageType']=='UncalAccel']
            gyro_df = imu_df[imu_df['MessageType']=='UncalGyro']
            mag_list.append(mag_df)
            acc_list.append(acc_df)
            gyro_list.append(gyro_df)

            if dir_name == 'train':
                gt_df = pd.read_csv(f'{path}/ground_truth.csv')
                gt_df = naming_collection_and_phone(gt_df, path)
                gt_list.append(gt_df)

        place = path.split('US-')[-1].split('/')[0]

        concated_gnss_df = pd.concat(gnss_list)
        concated_gnss_df.to_csv(f'{DATA_PATH}/shilver/{dir_name}/{place}_gnss.csv', index=False)

        concated_mag_df = pd.concat(mag_list)
        concated_acc_df = pd.concat(acc_list)
        concated_gyro_df = pd.concat(gyro_list)
        concated_mag_df.to_csv(f'{DATA_PATH}/shilver/{dir_name}/{place}_mag.csv', index=False)
        concated_acc_df.to_csv(f'{DATA_PATH}/shilver/{dir_name}/{place}_acc.csv', index=False)
        concated_gyro_df.to_csv(f'{DATA_PATH}/shilver/{dir_name}/{place}_gyro.csv', index=False)

        if dir_name == 'train':
            concated_gt_df = pd.concat(gt_list)
            concated_gt_df.to_csv(f'{DATA_PATH}/shilver/{dir_name}/{place}_gt.csv', index=False)

# %%
# generate train
generate_datasets(tr_paths, 'train')

# %%
# generate test
generate_datasets(te_paths, 'test')