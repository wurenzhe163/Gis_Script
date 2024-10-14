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
H5Dir = r'E:\SETP_ICESat-2数据\ATL_03\测试'
H5Oringin_Dir = r'E:\SETP_ICESat-2数据\ATL_03'
H5Paths = fsw.search_files(H5Dir,endwith='.h5')
Basenames = [os.path.basename(each) for each in fsw.search_files(H5Oringin_Dir,endwith='.h5')]

# 读取数据并融合
H5Datas_list = [pd.read_hdf(eachPath, key='df') for eachPath in tqdm(H5Paths,total=len(H5Paths))]

# 添加文件名列

H5_DFs = pd.concat(H5Datas_list, axis=0)

# 将多个信号置信度合并为一个列表
H5_DFs['signal_conf_ph'] = H5_DFs.apply(lambda row: [row['signal_conf_ph_1'],
                                                     row['signal_conf_ph_2'],
                                                     row['signal_conf_ph_3'],
                                                     row['signal_conf_ph_4'],
                                                     row['signal_conf_ph_5']], axis=1)

# 创建列 signal_conf_ph_gt_0，检查列表中是否有元素 > 0、>= 0
H5_DFs['signal_conf_ph_gt_0'] = H5_DFs['signal_conf_combined'].apply(lambda x: any(i > 0 for i in x))

# 时间转为年月日
H5_DFs.loc[:, 'date'] = H5_DFs['time'].apply(convert_delta_time)

# 保留有用的信息，删除无用的信息
H5_DFs[['time', 'date','lat', 'lon','height','subgroup','dist_ph_along','quality_ph','signal_conf_ph',
        'Sort', 'Area_pre']]

# %%
#-------------------------------------ATL06-------------------------------------