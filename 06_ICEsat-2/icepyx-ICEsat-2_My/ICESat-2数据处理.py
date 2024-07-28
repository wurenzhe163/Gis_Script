import geopandas as gpd
import pandas as pd
from tqdm import trange,tqdm
from shapely.geometry import Point,box

def filter_data_by_polygon(df, polygon_geo_df, batch_size=10000):
    '''求取SETP包含的激光点'''
    filtered_data_list = []
    for start in trange(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch_df = df.iloc[start:end].copy()
        batch_df['geometry'] = gpd.points_from_xy(batch_df['lon'], batch_df['lat'])
        batch_filtered_data = batch_df[gpd.GeoDataFrame(batch_df, geometry='geometry')\
                                       .within(polygon_geo_df.unary_union)]
        batch_filtered_data = batch_filtered_data.drop('geometry', axis=1)
        filtered_data_list.append(batch_filtered_data)
    filtered_data = pd.concat(filtered_data_list, ignore_index=True)
    return filtered_data

def df_to_shapefile(df, output_path, epsg=4326, batch_size=10000):
    '''将 DataFrame 转换为 shapefile'''
    if 'lon' not in df.columns or 'lat' not in df.columns:
        raise ValueError("DataFrame must contain 'lon' and 'lat' columns")
    
    # 初始化一个空的 GeoDataFrame，并设置几何列
    combined_geo_df = gpd.GeoDataFrame(columns=df.columns.tolist() + ['geometry'])
    combined_geo_df.set_geometry('geometry', inplace=True)
    combined_geo_df.set_crs(epsg=epsg, inplace=True)

    total_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
    df_iterator = (df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size))
    
    for batch_df in tqdm(df_iterator, total=total_batches, desc="Processing batches"):
        geometry = [Point(xy) for xy in zip(batch_df['lon'], batch_df['lat'])]
        batch_geo_df = gpd.GeoDataFrame(batch_df, geometry=geometry)
        batch_geo_df.set_crs(epsg=epsg, inplace=True)
        combined_geo_df = pd.concat([combined_geo_df, batch_geo_df], ignore_index=True)
    
    combined_geo_df.to_file(output_path, driver='ESRI Shapefile')



if __name__ == '__main__':
    combined_h5Path = r"D:\Dataset_and_Demo\ICESat-2\combined_data.h5"
    SETP_SHP = r'D:\BaiduSyncdisk\03_数据与总结\边界信息\SETP_Boundary\SETP_Boundary.shp'
    SETP_H5 = pd.read_hdf(combined_h5Path, key='df')
    SETP_DOM = gpd.read_file(SETP_SHP)
    SETP_BOUND = SETP_DOM.bounds
    SETP_BOUND_DOM = gpd.GeoDataFrame(SETP_BOUND, 
                                      geometry=[box(row.minx, row.miny, row.maxx, row.maxy) 
                                                for idx, row in SETP_BOUND.iterrows()], 
                                                crs=SETP_DOM.crs)

    # H5数据转为SHP点
    df_to_shapefile(SETP_H5.sample(100000),r'D:\Dataset_and_Demo\ICESat-2\SETP_ALT03_SAMPLE.shp') 
    
    # 过滤H5数据，求取SETP包含的激光点
    GL_H5 = filter_data_by_polygon(SETP_H5,SETP_DOM)
    df_to_shapefile(GL_H5,r'D:\Dataset_and_Demo\ICESat-2\GL_ALT03.shp')