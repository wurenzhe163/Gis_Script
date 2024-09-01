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
addnum = 10
file_list = fsw.search_files(r'E:\SETP_ICESat-2数据\ATL_03', '.h5')[addnum:addnum+10]

shapefile_path = r"D:\BaiduSyncdisk\02_论文相关\在写\SAM冰湖\数据\2023_05_31_to_2023_09_15_样本修正.shp"
gdf_polygons = gpd.read_file(shapefile_path)

# 假设原始的 CRS 是 EPSG:4326
original_crs = gdf_polygons.crs

# 转换为适当的投影坐标系，例如 EPSG:32633
projected_gdf = gdf_polygons.to_crs(epsg=32633)

# 应用100米的缓冲区
projected_gdf['geometry'] = projected_gdf.geometry.buffer(50)

# 将结果转换回原始的地理坐标系
gdf_polygons_buffered = projected_gdf.to_crs(original_crs)

# 构建空间索引
polygon_tree = STRtree(gdf_polygons_buffered.geometry)

def filter_data_by_polygon(df, polygon_tree, batch_size=10000):
    '''求取SETP包含的激光点'''
    filtered_data_list = []
    for start in trange(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch_df = df.iloc[start:end].copy()
        batch_df['geometry'] = gpd.points_from_xy(batch_df['lon'], batch_df['lat'])
        gdf_points = gpd.GeoDataFrame(batch_df, geometry='geometry')
        
        # 利用空间索引进行过滤
        possible_matches_index = polygon_tree.query(gdf_points.geometry, predicate='intersects')
        possible_matches_index = [item for sublist in possible_matches_index for item in sublist]  # 展平数组
        if possible_matches_index:
            possible_matches = gdf_polygons_buffered.iloc[possible_matches_index]
            print('length of possible_matches = {}'.format(len(possible_matches)))
            precise_matches = gdf_points[gdf_points.geometry.apply(lambda x: possible_matches.contains(x).any())]
            if not precise_matches.empty:
                precise_matches = precise_matches.drop(columns='geometry')
                filtered_data_list.append(precise_matches)
    
    if filtered_data_list:
        filtered_data = pd.concat(filtered_data_list, ignore_index=True)
        return filtered_data
    else:
        return pd.DataFrame()

# 提取所需信息，形成数据文件
for idx, file_path in enumerate(tqdm(file_list, desc="Processing Files")):
    combined_data = pd.DataFrame()
    data = h5py.File(file_path, 'r')
    for subgroup in tqdm(sub_file_list, desc="Processing Subgroups", leave=False):
        if subgroup in data:
            time_data = data.get(os.path.join(subgroup, 'heights/delta_time'))
            lat = data.get(os.path.join(subgroup, 'heights/lat_ph'))
            lon = data.get(os.path.join(subgroup, 'heights/lon_ph'))
            dist_ph_along = data.get(os.path.join(subgroup, 'heights/dist_ph_along'))
            height = data.get(os.path.join(subgroup, 'heights/h_ph'))
            signal_conf_ph = data.get(os.path.join(subgroup, 'heights/signal_conf_ph'))
            quality_ph = data.get(os.path.join(subgroup, 'heights/quality_ph'))

            if all(x is not None for x in [lat, lon, height, time_data, dist_ph_along, quality_ph, signal_conf_ph]):
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
                
                # 过滤数据，删除 signal_conf_ph_1、signal_conf_ph_2、signal_conf_ph_3、signal_conf_ph_4、signal_conf_ph_5 小于 0 的数据
                df = df[(df['signal_conf_ph_1'] >= 0) |
                        (df['signal_conf_ph_2'] >= 0) |
                        (df['signal_conf_ph_3'] >= 0) |
                        (df['signal_conf_ph_4'] >= 0) |
                        (df['signal_conf_ph_5'] >= 0)]
                
                # 过滤数据，根据多边形的包含关系
                filtered_df = filter_data_by_polygon(df, polygon_tree)
                
                if not filtered_df.empty:
                    # 连接属性表信息
                    gdf_filtered = gpd.GeoDataFrame(filtered_df, geometry=gpd.points_from_xy(filtered_df['lon'], filtered_df['lat']), crs=original_crs)
                    # 将多边形的属性（除去 'geometry' 列）连接到点数据中
                    joined_df = gpd.sjoin(gdf_filtered, gdf_polygons_buffered[['Sort', '关系', 'Area_pre', 'geometry']], how='left', predicate='intersects')
                    
                    if not joined_df.empty:
                        # 移除 'geometry' 列，以便合并到 combined_data 中
                        joined_df = joined_df.drop(columns='geometry')
                        combined_data = pd.concat([combined_data, joined_df], ignore_index=True)
    data.close()

    if not combined_data.empty:
        output_file_path = r'E:\SETP_ICESat-2数据\ATL_03\ATL03_SETPGL_ALL_{}.h5'.format(idx + addnum)
        combined_data.to_hdf(output_file_path, key='df', mode='w')

end_time = time.time()
print(f"Processing completed in {end_time - start_time:.2f} seconds")
