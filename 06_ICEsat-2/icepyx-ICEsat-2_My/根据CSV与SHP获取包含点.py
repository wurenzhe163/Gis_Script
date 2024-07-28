import pandas as pd
import geopandas as gpd
from shapely.geometry import Point  
from tqdm import tqdm

def filter_data_by_polygon(df, polygon_geo_df):
    df['geometry'] = gpd.points_from_xy(df['lon'], df['lat'])
    filtered_data = df[gpd.GeoDataFrame(df, geometry='geometry').within(polygon_geo_df.unary_union)]
    filtered_data = filtered_data.drop('geometry', axis=1)
    return filtered_data

shapefile_path = r'D:\Dataset_and_Demo\2023_05_31_to_2023_09_15_样本.shp'
point_path = r'D:\Dataset_and_Demo\ICESat-2\text_all.csv'
gdf_polygons = gpd.read_file(shapefile_path)
df = pd.read_csv(point_path) 
geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
# 创建GeoDataFrame
gdf_points = gpd.GeoDataFrame(df, geometry=geometry)
del df

# 设置坐标系（假设shapefile和数据点使用相同的坐标系，例如WGS84）
gdf_points.set_crs('EPSG:4326', inplace=True)

# 重置索引
gdf_points = gdf_points.reset_index(drop=True)

# # 保存为GeoJSON
# gdf_points.to_file(r'D:\Dataset_and_Demo\ICESat-2\SETP_ATL03.geojson', driver='GeoJSON', index=False)

# # 进行空间连接，将点数据与shapefile中的多边形关联
# gdf_joined = gpd.sjoin(gdf_points, gdf_polygons, how="inner",predicate='intersects')

# 分批次操作
batch_size = 10000
n_batches = (len(gdf_points) + batch_size - 1) // batch_size  # 计算总批次数
result = []

for n in tqdm(range(n_batches), desc="Processing batches"):
    gdf_batch = gdf_points[n * batch_size:(n + 1) * batch_size]
    gdf_joined = gpd.sjoin(gdf_batch, gdf_polygons, how="inner", predicate='intersects')
    if len(gdf_joined) != 0:
        result.append(gdf_joined)

gdf_result = gpd.GeoDataFrame(pd.concat(result, ignore_index=True))

# 保存结果（可选）
output_path = r'D:\Dataset_and_Demo\ICESat-2\SETP_ATL03.geojson'
gdf_result.to_file(output_path)