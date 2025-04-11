# %%
import icepyx as ipx
import os
from pprint import pprint
from PackageDeepLearn.utils import file_search_wash as fsw


# %%
beam_list = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
var_list = ['h_ph', 'lat_ph', 'lon_ph', 'quality_ph', 'signal_conf_ph']
START_DATE = '2023-01-01'
END_DATE = '2023-01-31'
short_name = 'ATL03' # ATL06、ATL08
spatial_extent = r'D:\BaiduSyncdisk\03_数据与总结\边界信息\SETP_Boundary\SETP_Boundary.shp'
# spatial_extent = [(-55, 68), (-55, 71), (-48, 71), (-48, 68), (-55, 68)]
date_range = {"start_date": START_DATE, "end_date": END_DATE}

# %%
# Specify the version or use the latest version
region_a = ipx.Query(short_name, spatial_extent, date_range, version='006')
# 设置子集参数
# region_a.order_vars.remove(all=True)
# region_a.order_vars.append(beam_list=beam_list, var_list=var_list)
# region_a.subsetparams(Coverage=region_a.order_vars.wanted)
path = r'D:\Dataset_and_Demo\ICESat-2\2023_ATL03_3'
region_a.download_granules(path)

# %%
short_name = 'ATL06'
date_range = {"start_date": START_DATE, "end_date": END_DATE}
region_a = ipx.Query(short_name, spatial_extent, date_range, version='006')
path = r'D:\Dataset_and_Demo\ICESat-2\2023_ATL06'
region_a.download_granules(path) 

# %%
short_name = 'ATL08'
date_range = {"start_date": START_DATE, "end_date": END_DATE}
region_a = ipx.Query(short_name, spatial_extent, date_range, version='006')
path = r'D:\Dataset_and_Demo\2023_ATL08'
region_a.download_granules(path) 








latlims = [28.516,28.520]
lonlims = [97.5039,97.5080]

# 定义四个角点
top_left = (lonlims[0], latlims[1])
top_right = (lonlims[1], latlims[1])
bottom_right = (lonlims[1], latlims[0])
bottom_left = (lonlims[0], latlims[0])

# 构建spatial_extent列表
spatial_extent = [top_left, top_right, bottom_right, bottom_left, top_left]


# beam_list = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
beam_list = ['gt3r']
var_list = ['h_ph', 'lat_ph', 'lon_ph', 'quality_ph', 'signal_conf_ph']
START_DATE = '2023-01-01'
END_DATE = '2023-12-31'
short_name = 'ATL03' # ATL06、ATL08
date_range = {"start_date": START_DATE, "end_date": END_DATE}

region_a = ipx.Query(short_name, spatial_extent, date_range, version='006')

region_a.avail_granules(ids=True, cycles=False, tracks=False,cloud=False)