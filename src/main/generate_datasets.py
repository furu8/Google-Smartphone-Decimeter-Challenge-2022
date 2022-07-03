# %%
import numpy as np 
import pandas as pd
import glob as gb
from tqdm import tqdm
DATA_PATH = 'D:/OpenData/Kaggle/smartphone-decimeter-2022'

trains = gb.glob(f"{DATA_PATH}/bronze/train/*/*/*")
tests = gb.glob(f"{DATA_PATH}/bronze/test/*/*/*")

places = [
    'LAX-1', 'LAX-2', 'LAX-3', 'LAX-5', 'MTV-1', 'MTV-2', 'MTV-3', 'OAK-1', 'OAK-2',
 'SFO-1', 'SFO-2', 'SJC-1', 'SJC-2', 'SVL-1', 'SVL-2'
]
# %%
# pathの順序を走行場所ごとになるように変更
def make_paths_each_place(paths):
    new_paths = []
    for place in places:
        print(place)
        # if 'SFO-1' == place: #特定のファイルのみ変更したいときに利用
        tmp_paths = []
        for path in paths:
            prev_place_path = path.split('US-')[0] # 走行場所より前のpath
            place_name = path.split('US-')[-1].split('\\')[0] # 走行場所抽出
            next_place_path = path.split('\\')[2:][0] # 走行場所より後のpath
            if place == place_name: 
                tmp_paths.append(prev_place_path+'US-'+place_name+'/'+next_place_path+'/')
        if tmp_paths == []:
            continue
        new_paths.append(tmp_paths)
    return new_paths

# %%
tr_paths = [np.unique(np.array(places_path)) for places_path in make_paths_each_place(trains)]
te_paths = [np.unique(np.array(places_path)) for places_path in make_paths_each_place(tests)]
tr_paths

# %%
# 結合の識別元
def naming_collection_and_phone(df, path):
    new_df = df.copy()
    new_df['collectionName'] = path.split('\\')[-1].split('/')[0]
    new_df['phoneName'] = path.split('/')[-2]
    return new_df

def generate_datasets(paths, dir_name):
    for phone_paths in tqdm(paths):
        concated_gnss_df = pd.DataFrame()
        concated_mag_df = pd.DataFrame()
        concated_acc_df = pd.DataFrame()
        concated_gyro_df = pd.DataFrame()
        concated_gt_df = pd.DataFrame()
        for path in phone_paths:

            gnss_df = pd.read_csv(f'{path}/device_gnss.csv')
            gnss_df = naming_collection_and_phone(gnss_df, path)
            concated_gnss_df = pd.concat([concated_gnss_df, gnss_df])

            imu_df = pd.read_csv(f'{path}/device_imu.csv')
            imu_df = naming_collection_and_phone(imu_df, path)
            mag_df = imu_df[imu_df['MessageType']=='UncalMag']
            acc_df = imu_df[imu_df['MessageType']=='UncalAccel']
            gyro_df = imu_df[imu_df['MessageType']=='UncalGyro']
            concated_mag_df = pd.concat([concated_mag_df, mag_df])
            concated_acc_df = pd.concat([concated_acc_df, acc_df])
            concated_gyro_df = pd.concat([concated_gyro_df, gyro_df])

            if dir_name == 'train':
                gt_df = pd.read_csv(f'{path}/ground_truth.csv')
                gt_df = naming_collection_and_phone(gt_df, path)
                concated_gt_df = pd.concat([concated_gt_df, gt_df])
 
        if concated_gnss_df.empty:
            print(phone_paths)
            break
        place = path.split('US-')[-1].split('/')[0]

        concated_gnss_df.to_csv(f'{DATA_PATH}/shilver/{dir_name}/{place}_gnss.csv', index=False)
        
        concated_mag_df.to_csv(f'{DATA_PATH}/shilver/{dir_name}/{place}_mag.csv', index=False)
        concated_acc_df.to_csv(f'{DATA_PATH}/shilver/{dir_name}/{place}_acc.csv', index=False)
        concated_gyro_df.to_csv(f'{DATA_PATH}/shilver/{dir_name}/{place}_gyro.csv', index=False)

        if dir_name == 'train':
            concated_gt_df.to_csv(f'{DATA_PATH}/shilver/{dir_name}/{place}_gt.csv', index=False)

# %%
# generate train
generate_datasets(tr_paths, 'train')

# %%
# generate test
generate_datasets(te_paths, 'test')

