import ee
import geemap
import os
import json

geemap.set_proxy(port=10809)
ee.Initialize()
ee.Authenticate()
  
#--------------------------预加载冰湖数据,测试的时候加上Filter_bound
Glacial_lake = ee.FeatureCollection('projects/ee-mrwurenzhe/assets/Glacial_lake/2023_05_31_to_2023_09_15_SpatialJoin').map(
    lambda feature: feature.set('ID', ee.Number.parse(feature.get('ID')))
).sort('ID')

# 计算geometry、质心点、最小包络矩形
Geo_ext = lambda feature: feature.set({
                                    'Geo': feature.geometry(),
                                    'Centroid': feature.geometry().centroid(),
                                    'Rectangle': feature.geometry().bounds()})

Glacial_lake_C = Glacial_lake.map(Geo_ext)
Num_list = Glacial_lake.size().getInfo()
Glacial_lake_A_GeoList = Glacial_lake.toList(Num_list)
Glacial_lake_C_CentriodList = ee.List(Glacial_lake_C.reduceColumns(ee.Reducer.toList(),['Centroid']).get('list'))
Glacial_lake_R_RectangleList = ee.List(Glacial_lake_C.reduceColumns(ee.Reducer.toList(),['Rectangle']).get('list'))



for i in range(Num_list):
    AOI_Bound = ee.Feature(Glacial_lake_R_RectangleList.get(i)).geometry()
    aoi_info_path = os.path.join(r'E:\Dataset_and_Demo\SETP_GL\JSONs', "{}_ADMeanFused_AOI.json".format(f'{i:05d}'))
    with open(aoi_info_path, 'w') as f:
        json.dump(AOI_Bound.getInfo(), f)