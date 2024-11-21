# %%
from PackageDeepLearn.utils import file_search_wash as fsw
import timescale.time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
warnings.filterwarnings('ignore')

def convert_delta_time(delta_time, gps_epoch=1198800018.0):
    delta_time = np.atleast_1d(delta_time)
    gps_seconds = gps_epoch + delta_time
    time_leaps = timescale.time.count_leap_seconds(gps_seconds)
    time_julian = 2400000.5 + timescale.time.convert_delta_time(
        gps_seconds - time_leaps, epoch1=(1980,1,6,0,0,0),
        epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0)
    Y, M, D, h, m, s = timescale.time.convert_julian(time_julian, format='tuple')
    Y, M, D = int(Y[0]), int(M[0]), int(D[0])
    return f'{Y}-{M:02d}-{D:02d}'
# %%
#-------------------------------------ATL03-------------------------------------
H5Dir = r'G:\SETP_ICESat-2\ATL_03_GlobalGeolocatedPhoton\ATL03_Noise\test'
H5Paths = fsw.search_files(H5Dir,endwith='.h5')

# 读取数据并融合
H5Datas_list = [pd.read_hdf(eachPath, key='df') for eachPath in tqdm(H5Paths,total=len(H5Paths))]

# 添加文件名列
H5_DFs = pd.concat(H5Datas_list, axis=0)

# 将多个信号置信度合并为一个列表
H5_DFs['signal_conf_combined'] = H5_DFs.apply(lambda row: [ row['signal_conf_ph_1'],
                                                            row['signal_conf_ph_2'],
                                                            row['signal_conf_ph_3'],
                                                            row['signal_conf_ph_4'],
                                                            row['signal_conf_ph_5']], axis=1)

# 创建列 signal_conf_ph_gt_0，检查列表中是否有元素 > 0、>= 0
# H5_DFs['signal_conf_ph_Any'] = H5_DFs['signal_conf_combined'].apply(lambda x: any(i > 0 for i in x))
H5_DFs['signal_conf_ph_Water'] = False; H5_DFs.loc[H5_DFs['signal_conf_combined'].apply(lambda x: x[-1] > 1), 'signal_conf_ph_Water'] = True


# 时间转为年月日
unique_times = H5_DFs['time'].unique()
converted_times = {time: convert_delta_time(time) for time in unique_times}
H5_DFs['date'] = H5_DFs['time'].map(converted_times)

# 保留有用的信息，删除无用的信息
H5_DFs = H5_DFs[['time', 'date','lat', 'lon','height','dist_ph_along','quality_ph','signal_conf_combined','signal_conf_ph_Water','subgroup',
        'Sort', 'Area_pre']]

H5_DFs.to_hdf(os.path.join(H5Dir,'ATL03_ALL.h5'), key='df', mode='w',index=False)
H5_DFs[H5_DFs['signal_conf_ph_Water'] == True].to_hdf(os.path.join(H5Dir,'ATL03_Water.h5'), key='df', mode='w',index=False)

# %%
#-------------------------------------ATL06-------------------------------------
H5Dir = r'G:\SETP_ICESat-2\ATL_06_Landice\ATL06_ALL'
H5Paths = fsw.search_files(H5Dir,endwith='.h5')

H5Datas_list = [pd.read_hdf(eachPath, key='df') for eachPath in tqdm(H5Paths,total=len(H5Paths))]
H5_DFs = pd.concat(H5Datas_list, axis=0)

# 时间转为年月日
unique_times = H5_DFs['time'].unique()
converted_times = {time: convert_delta_time(time) for time in unique_times}
H5_DFs['date'] = H5_DFs['time'].map(converted_times)

H5_DFs = H5_DFs[['time', 'date','lat', 'lon','height','signal_conf_ph','id','subgroup','Sort', 'Area_pre']]
H5_DFs.to_hdf(os.path.join(H5Dir,'ATL06_ALL.h5'), key='df', mode='w',index=False)

# %%
#-------------------------------------ATL08-------------------------------------
H5Dir = r'G:\SETP_ICESat-2\ATL_08_LandVegetation\ATL08_ALL'
H5Paths = fsw.search_files(H5Dir,endwith='.h5')

# 读取数据并融合
H5Datas_list = [pd.read_hdf(eachPath, key='df') for eachPath in tqdm(H5Paths,total=len(H5Paths))]

# 添加文件名列
H5_DFs = pd.concat(H5Datas_list, axis=0)

# 时间转为年月日
unique_times = H5_DFs['time'].unique()
converted_times = {time: convert_delta_time(time) for time in unique_times}
H5_DFs['date'] = H5_DFs['time'].map(converted_times)

H5_DFs = H5_DFs[['time','date','lat', 'lon','height_centroid','height_canopy','dem','id','cloud','subgroup','Sort', 'Area_pre']]
H5_DFs.to_hdf(os.path.join(H5Dir,'ATL08_ALL.h5'), key='df', mode='w',index=False)

# %%
#-------------------------------------ATL13-------------------------------------
H5Dir = r'G:\SETP_ICESat-2\ATL_13_InlandSurfaceWaterData\ATL13_ALL'
H5Paths = fsw.search_files(H5Dir,endwith='.h5')
# 读取数据并融合
H5Datas_list = [pd.read_hdf(eachPath, key='df') for eachPath in tqdm(H5Paths,total=len(H5Paths))]
# 添加文件名列
H5_DFs = pd.concat(H5Datas_list, axis=0)

# 时间转为年月日
unique_times = H5_DFs['time'].unique()
converted_times = {time: convert_delta_time(time) for time in unique_times}
H5_DFs['date'] = H5_DFs['time'].map(converted_times)

H5_DFs = H5_DFs[['time','date','lat', 'lon','segment_lat','segment_lon','height_surface','water_depth','dem',
                 'id','cloud', 'ice_flag','inland_water_body_type','subgroup','Sort', 'Area_pre']]

H5_DFs.to_hdf(os.path.join(H5Dir,'ATL13_ALL.h5'), key='df', mode='w',index=False)