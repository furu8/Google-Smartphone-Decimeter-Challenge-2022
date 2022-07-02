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
def generate_datasets(paths, dir_name):
    gnss_list, imu_list, gt_list = [], [], []
    for phone_paths in tqdm(paths):
        for path in phone_paths:
            gnss_list.append(pd.read_csv(f'{path}/device_gnss.csv'))
            imu_list.append(pd.read_csv(f'{path}/device_imu.csv'))
            if dir_name == 'train':
                gt_list.append(pd.read_csv(f'{path}/ground_truth.csv'))

        place = path.split('US-')[-1].split('/')[0]

        gnss_df = pd.concat(gnss_list)
        gnss_df.to_csv(f'{DATA_PATH}/shilver/{dir_name}/{place}_gnss.csv', index=False)
        imu_df = pd.concat(imu_list)
        imu_df.to_csv(f'{DATA_PATH}/shilver/{dir_name}/{place}_imu.csv', index=False)
        if dir_name == 'train':
            gt_df = pd.concat(gt_list)
            gt_df.to_csv(f'{DATA_PATH}/shilver/{dir_name}/{place}_gt.csv', index=False)

# %%
# generate train
generate_datasets(tr_paths, 'train')

# %%
# generate test
generate_datasets(te_paths, 'test')