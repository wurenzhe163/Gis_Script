import os
import h5py
import pandas as pd
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import geopandas as gpd
from PackageDeepLearn.utils import file_search_wash as fsw
# 所有条带
sub_file_list = ['gt1l/', 'gt1r/', 'gt2l/', 'gt2r/', 'gt3l/', 'gt3r/']
combined_data = pd.DataFrame()
start_time = time.time()
file_list = fsw.search_files(r'D:\Dataset_and_Demo\ICESat-2\2023_ATL03_2','.h5')

# 提取所需信息，形成数据文件
for file_path in tqdm(file_list, desc="Processing Files"):
    data = h5py.File(file_path, 'r')
    for subgroup in tqdm(sub_file_list, desc="Processing Subgroups", leave=False):
        if subgroup in data:
            time_data = data.get(os.path.join(subgroup, 'heights/delta_time'))
            lat = data.get(os.path.join(subgroup, 'heights/lat_ph'))
            lon = data.get(os.path.join(subgroup, 'heights/lon_ph'))
            dist_ph_along = data.get(os.path.join(subgroup, 'heights/dist_ph_along'))
            height = data.get(os.path.join(subgroup, 'heights/h_ph'))
            signal_conf_ph = data.get(os.path.join(subgroup, 'heights/signal_conf_ph'))
            quality_ph    =  data.get(os.path.join(subgroup, 'heights/quality_ph'))

            if all(x is not None for x in [lat, lon, height, time_data, dist_ph_along,quality_ph,signal_conf_ph]):
                df = pd.DataFrame(data={
                    'time': time_data[:],
                    'lat': lat[:],
                    'lon': lon[:],
                    'dist_ph_along': dist_ph_along[:],
                    'height': height[:],
                    'quality_ph': quality_ph[:],
                    'signal_conf_ph_1': signal_conf_ph[:, 0],
                    'signal_conf_ph_2': signal_conf_ph[:, 1],
                    'signal_conf_ph_3': signal_conf_ph[:, 2],
                    'signal_conf_ph_4': signal_conf_ph[:, 3],
                    'signal_conf_ph_5': signal_conf_ph[:, 4]
                })
                df['subgroup'] = subgroup

                combined_data = pd.concat([combined_data, df], ignore_index=True)
    data.close()
combined_data.to_hdf(r'D:\Dataset_and_Demo\combined_data.h5', key='df', mode='w')
# combined_data.to_csv(r'D:\Dataset_and_Demo\text_all.csv', index=True)
end_time = time.time()
print(f"Processing completed in {end_time - start_time:.2f} seconds")

