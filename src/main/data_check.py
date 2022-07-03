# %% [markdown]
# - サンプル確認
# - ディレクトリ（走行場所）の一覧
# - スマホの名前一覧
# - millis確認
# - gt確認
# %%
import numpy as np 
import pandas as pd
import glob as gb
DATA_PATH = 'D:/OpenData/Kaggle/smartphone-decimeter-2022'
# %%
def read_df(stage, filename):
    return pd.read_csv(f"{DATA_PATH}/{stage}/{filename}.csv")

# %%
# サンプル確認
sub = read_df('submission', 'sample_submission')
sub.head()

# %%
# ディレクトリ（走行場所）の一覧
def check_place(filname):
    places = []
    for fname in filname:
        places.append(fname.split('US-')[-1])
    return np.unique(np.array(places))  

train = gb.glob(f"{DATA_PATH}/bronze/train/*")
test = gb.glob(f"{DATA_PATH}/bronze/test/*")
all_filename = train+test

print('all:', np.sort(check_place(all_filename)))
print('train:', np.sort(check_place(train)))
print('test:', np.sort(check_place(test)))
print('diff:', np.sort(np.array(list(set(check_place(train)) - set(check_place(test))))))

# %%
# スマホの名前一覧
def check_phone(filname):
    phones = []
    for fname in filname:
        phones.append(fname.split('\\')[-1])
    return np.unique(np.array(phones))

train = gb.glob(f"{DATA_PATH}/bronze/train/*/*")
test = gb.glob(f"{DATA_PATH}/bronze/test/*/*")
all_filename = train+test
print('all:', np.sort(check_phone(all_filename)))
print('train:', np.sort(check_phone(train)))
print('test:', np.sort(check_phone(test)))
print('tr-te->diff:', np.sort(np.array(list(set(check_phone(train)) - set(check_phone(test))))))
print('te-tr->diff:', np.sort(np.array(list(set(check_phone(test)) - set(check_phone(train))))))
# %%
# 走行場所ごとのCSVファイル数
trains = gb.glob(f"{DATA_PATH}/bronze/train/*/*/*")
tests = gb.glob(f"{DATA_PATH}/bronze/test/*/*/*")

places = [
    'LAX-1', 'LAX-2', 'LAX-3', 'LAX-5', 'MTV-1', 'MTV-2', 'MTV-3', 'OAK-1', 'OAK-2',
 'SFO-1', 'SFO-2', 'SJC-1', 'SJC-2', 'SVL-1', 'SVL-2'
]

filenames = [
    'device_gnss',
    'device_imu',
    'ground_truth'
]

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

tr_paths = [np.unique(np.array(places_path)) for places_path in make_paths_each_place(trains)]
te_paths = [np.unique(np.array(places_path)) for places_path in make_paths_each_place(tests)]

# generate
def generate_datasets(paths):
    gnss_list, imu_list, gt_list = [], [], []
    for place, path in zip(places, paths):
        gnss_list.append(place)
        imu_list.append(len(path))
    df = pd.DataFrame()
    df['place'] = gnss_list
    df['csvfile_num'] = imu_list

    display(df)

generate_datasets(tr_paths)
generate_datasets(te_paths)