# %%
import numpy as np 
import pandas as pd
import glob as gb
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly
import plotly.express as px
DATA_PATH = 'D:/OpenData/Kaggle/smartphone-decimeter-2022'

def visualize_trafic(df, center, zoom=9):
    fig = px.scatter_mapbox(df,
                            # Here, plotly gets, (x,y) coordinates
                            lat="LatitudeDegrees",
                            lon="LongitudeDegrees",
                            #Here, plotly detects color of series
                            color="phoneName",
                            labels="phoneName",
                            zoom=zoom,
                            center=center,
                            height=600,
                            width=800)
    fig.update_layout(mapbox_style='stamen-terrain')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    fig.show()

def visualize_collection(df, collection):
    target_df = df[df['collectionName']==collection].copy()
    lat_center = target_df['LatitudeDegrees'].mean()
    lng_center = target_df['LongitudeDegrees'].mean()
    center = {"lat":lat_center, "lon":lng_center}
    visualize_trafic(target_df, center)

def read_gt(place):
    return pd.read_csv(f'{DATA_PATH}/shilver/train/{place}_gt.csv')

def read_pred(place, name):
    df = pd.read_csv(f'{DATA_PATH}/submission/{name}.csv')
    return df[df['tripId'].str.contains(place)]

# %%
def all_place_visualize(places, name=None):
    for place in places:
        if name is None:
            df = read_gt(place)
        else:
            df = read_pred(place, name)
            df['collectionName'] = df['tripId'].str.split('/', expand=True)[0]
            df['phoneName'] = df['tripId'].str.split('/', expand=True)[1]
        
        print(len(df['collectionName'].unique()))
        for collection in df['collectionName'].unique():
            print(collection)
            visualize_collection(df, collection)

# %%[markdown]
# ## train
"""各placeのデータ数参考
place	csvfile_num
LAX-1       8
LAX-2       8
LAX-3       4
LAX-5       4
MTV-1       72
MTV-2       42
MTV-3       4
OAK-1       0
OAK-2       0
SFO-1       4
SFO-2       4
SJC-1       10
SJC-2       4
SVL-1       4
SVL-2       2
"""

# %%
tr_places = [
    # 'LAX-1',
    # 'LAX-2', # testにない
    # 'LAX-3',
    # 'LAX-5',
    # 'MTV-1',
    # 'MTV-2',
    # 'MTV-3', # testにない
    # 'SFO-1', # testにない
    # 'SFO-2', # testにない
    # 'SJC-1',
    # 'SJC-2',
    # 'SVL-1', # testにない
    # 'SVL-2'
]
# %%
# train
all_place_visualize(tr_places)

# %%[markdown]
# ## test
"""各placeのデータ数参考
place	csvfile_num
LAX-1	    4
LAX-2	    0
LAX-3	    4
LAX-5	    2
MTV-1	    14
MTV-2	    2
MTV-3	    0
OAK-1	    2
OAK-2	    2
SFO-1	    0
SFO-2	    0
SJC-1	    4
SJC-2	    1
SVL-1	    0
SVL-2	    1
"""

# %%
te_places = [
    # 'LAX-1',
    # 'LAX-3',
    # 'LAX-5',
    # 'MTV-1',
    # 'MTV-2',
    # 'OAK-1',
    # 'OAK-2',
    # 'SJC-1',
    # 'SJC-2',
    # 'SVL-2'
]

# %%
# baseline_saito
all_place_visualize(te_places, 'baseline_saito')

# %%
