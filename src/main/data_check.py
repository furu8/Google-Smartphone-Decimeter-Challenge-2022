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
print('diff:', np.sort(np.array(list(set(check_phone(train)) - set(check_phone(test))))))
# %%
# millis確認
