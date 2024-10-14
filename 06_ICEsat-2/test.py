import os
import h5py
import pandas as pd
from tqdm import tqdm, trange
import time
import geopandas as gpd
from shapely.strtree import STRtree
from shapely.geometry import Point
from PackageDeepLearn.utils import file_search_wash as fsw

# 所有条带
sub_file_list = ['gt1l/', 'gt1r/', 'gt2l/', 'gt2r/', 'gt3l/', 'gt3r/']
start_time = time.time()
addnum = 0
file_list = fsw.search_files(r'E:\SETP_ICESat-2数据\ATL_03', '.h5')[0:65]

shapefile_path = r"D:\BaiduSyncdisk\02_论文相关\在写\SAM冰湖\数据\2023_05_31_to_2023_09_15_样本修正.shp"
gdf_polygons = gpd.read_file(shapefile_path)

# 假设原始的 CRS 是 EPSG:4326
original_crs = gdf_polygons.crs

# 转换为适当的投影坐标系，例如 EPSG:32633
projected_gdf = gdf_polygons.to_crs(epsg=32633)

# 应用100米的缓冲区
projected_gdf['geometry'] = projected_gdf.geometry.buffer(50)
def process_spatial_join(gdf_batch, gdf_polygons_buffered):
    # 执行空间连接
    joined_df = gpd.sjoin(gdf_batch, gdf_polygons_buffered, how='left', predicate='intersects')

    # 确保只有那些与目标几何体相交的条目被保留
    if 'index_right' in joined_df.columns:
        joined_df = joined_df.dropna(subset=['index_right'])

    return joined_df

# 将结果转换回原始的地理坐标系
gdf_polygons_buffered = projected_gdf.to_crs(original_crs)

file_list = fsw.search_files(r'E:\SETP_ICESat-2数据\ATL_13_InlandSurfaceWaterData', '.h5')
ATL13_NoData = []
# 提取所需信息，形成数据文件
for idx, file_path in enumerate(tqdm(file_list, desc="Processing Files")):
    combined_data = pd.DataFrame()
    data = h5py.File(file_path, 'r')
    output_file_path = r'E:\SETP_ICESat-2数据\ATL_13_InlandSurfaceWaterData\ATL13_ALL\{}_{}.h5'.format(os.path.basename(file_path).split('.')[0],idx + addnum)
    if os.path.exists(output_file_path):
        print('{} 存在，跳过'.format(output_file_path))
        continue
    for subgroup in tqdm(sub_file_list, desc="Processing Subgroups", leave=False):
        if subgroup in data:
            time_data = data.get(os.path.join(subgroup, 'delta_time'))
            lat = data.get(os.path.join(subgroup, 'bottom_lat'))
            segment_lat = data.get(os.path.join(subgroup, 'segment_lat'))
            lon = data.get(os.path.join(subgroup, 'bottom_lon'))
            segment_lon = data.get(os.path.join(subgroup, 'segment_lon'))
            height_surface = data.get(os.path.join(subgroup, 'ht_water_surf'))
            water_depth = data.get(os.path.join(subgroup, 'water_depth'))
            dem = data.get(os.path.join(subgroup, 'segment_dem_ht'))
            id = data.get(os.path.join(subgroup, 'segment_id_beg'))
            # Cloud probability from ASR.; (Meanings: [0 1 2 3 4 5]) 
            # (Values: ['clear_with_high_confidence', 'clear_with_medium_confidence', 'clear_with_low_confidence', 
            # 'cloudy_with_low_confidence', 'cloudy_with_medium_confidence', 'cloudy_with_high_confidence'])
            cloud = data.get(os.path.join(subgroup, 'cloud_flag_asr_atl09')) 
            ice_flag = data.get(os.path.join(subgroup, 'ice_flag'))
            inland_water_body_type = data.get(os.path.join(subgroup, 'inland_water_body_type'))

            if all(x is not None for x in [time_data, lat,segment_lat, lon,segment_lon,height_surface,water_depth, dem ,id,cloud,ice_flag,inland_water_body_type]):
                # print([len(i) for i in [time_data, lat,segment_lat, lon,segment_lon,height_surface,water_depth, dem ,id,cloud,ice_flag,inland_water_body_type]])
                df = pd.DataFrame(data={
                    'time': time_data[:],
                    'lat': lat[:],
                    'segment_lat': segment_lat[:],
                    'lon': lon[:],
                    'segment_lon': segment_lon[:],
                    'height_surface': height_surface[:],
                    'water_depth': water_depth[:],
                    'dem': dem[:],
                    'id':id[:],
                    'cloud':cloud[:],
                    'ice_flag':ice_flag[:],
                    'inland_water_body_type':inland_water_body_type[:]
                })
                df['subgroup'] = subgroup

                if not df.empty:
                    # 连接属性表信息
                    gdf_filtered = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs=original_crs)
                    
                    batch_size = 10000
                    n_batches = (len(gdf_filtered) + batch_size - 1) // batch_size  # 计算总批次数
                    result = []

                    for n in tqdm(range(n_batches), desc="Processing batches"):
                        gdf_batch = gdf_filtered[n * batch_size:(n + 1) * batch_size]
                        joined_df = process_spatial_join(gdf_batch, gdf_polygons_buffered)
                        if len(joined_df)>0:
                            pass

                        if not joined_df.empty:
                            # 移除 'geometry' 列，以便合并到 combined_data 中
                            joined_df = joined_df.drop(columns='geometry')
                            combined_data = pd.concat([combined_data, joined_df], ignore_index=True)
    data.close()

    if not combined_data.empty:
        combined_data.to_hdf(output_file_path, key='df', mode='w')
    else:
        print(f"No data found in {file_path}")
        ATL13_NoData.append(file_path)
