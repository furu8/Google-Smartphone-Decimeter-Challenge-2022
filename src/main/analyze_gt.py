# %%
import numpy as np 
import pandas as pd
import glob as gb
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly
import plotly.express as px
DATA_PATH = 'D:/OpenData/Kaggle/smartphone-decimeter-2022'

# %%
def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    """
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    dist = 2 * RADIUS * np.arcsin(a**0.5)
    return dist

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

# %%
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
places = [
    'LAX-1',
    'LAX-2', # testにない
    'LAX-3',
    'LAX-5',
    'MTV-1',
    'MTV-2',
    'MTV-3', # testにない
    'SFO-1', # testにない
    'SFO-2', # testにない
    'SJC-1',
    'SJC-2',
    'SVL-1', # testにない
    'SVL-2'
]
# %%
def all_place_visualize():
    for place in places:
        df = read_gt(place)
        print(len(df['collectionName'].unique()))
        for collection in df['collectionName'].unique():
            print(collection)
            visualize_collection(df, collection)
# %%
all_place_visualize()
