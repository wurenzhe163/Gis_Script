import ee
import geemap
import sys,os
sys.path.append(r'D:\09_Code\Gis_Script\GEE_Func')
from S1_distor_dedicated import load_S1collection,S1_CalDistor,DEM_caculator
from GEE_DataIOTrans import BandTrans,DataTrans,DataIO,Vector_process
Eq_pixels = DataTrans.Eq_pixels
from GEE_CorreterAndFilters import S1Corrector
from GEEMath import get_minmax
import math
from functools import partial
from tqdm import tqdm 

# 重载函数
# import importlib
# importlib.reload(S1_distor_dedicated)

geemap.set_proxy(port=10809)
ee.Authenticate()
ee.Initialize()


import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm

def Line2Points(feature,region,scale=30):
    '''经度小的会作为start_point，因此Ascedning左到右，Descending左到右'''
    # 从Feature中提取线几何对象
    line_geometry = ee.Feature(feature).geometry().intersection(region, maxError=1)
    
    # 获取线的所有坐标点
    coordinates = line_geometry.coordinates()
    start_point = ee.List(coordinates.get(0)) # 起点坐标
    end_point = ee.List(coordinates.get(-1))  # 终点坐标
    
    # 获取线段的总长度
    length = line_geometry.length()

    # 计算插入点的数量
    num_points = length.divide(scale).floor()  # .subtract(1)

    # 计算每个间隔点的坐标
    def interpolate(i):
        i = ee.Number(i)
        fraction = i.divide(num_points)
        interpolated_lon = ee.Number(start_point.get(0)).add(
            ee.Number(end_point.get(0)).subtract(ee.Number(start_point.get(0))).multiply(fraction))
        interpolated_lat = ee.Number(start_point.get(1)).add(
            ee.Number(end_point.get(1)).subtract(ee.Number(start_point.get(1))).multiply(fraction))
        return ee.Feature(ee.Geometry.Point([interpolated_lon, interpolated_lat]))

    # 使用条件表达式过滤长度为0的线段
    filtered_points = ee.FeatureCollection(ee.Algorithms.If(num_points.gt(0), 
                                                            ee.FeatureCollection(ee.List.sequence(1, num_points).map(interpolate)),
                                                            ee.FeatureCollection([])))

    return filtered_points

def filter_Listlen(item,len=3):
    '''将list长度小于3的过滤，该函数len=3起步判断，由于len=2的时候会将单点判定为正确，故存在误差为了节省计算量，未对此处进行修改'''
    item_list = ee.List(item)
    return ee.Algorithms.If(item_list.size().gte(len), item_list, None)

def get_neighborhood_info(point,image,scale=15,reduceSacle=30,neighborhood_type='4',Pointfilter=False):
    
    point = ee.Geometry.Point(point)
    buffer_region = point.buffer(scale)

    # 生成矩形区域
    region = buffer_region.bounds()
    # 获取矩形的坐标
    coords  = ee.List(region.coordinates().get(0))

    # 提取四个角点坐标
    corner1 = ee.Geometry.Point(ee.List(coords.get(0)))
    corner2 = ee.Geometry.Point(ee.List(coords.get(1)))
    corner3 = ee.Geometry.Point(ee.List(coords.get(2)))
    corner4 = ee.Geometry.Point(ee.List(coords.get(3)))
    
    if neighborhood_type=='4':
        # 创建包含所有点的MultiPoint几何对象
        region_coord = ee.Geometry.MultiPoint([
                                            corner1.coordinates(), corner2.coordinates(), 
                                            corner3.coordinates(), corner4.coordinates()])
    
    elif neighborhood_type=='9': 
        # 创建线段并计算边的中心点
        edge_center1 = ee.Geometry.LineString([corner1.coordinates(), corner2.coordinates()]).centroid()
        edge_center2 = ee.Geometry.LineString([corner2.coordinates(), corner3.coordinates()]).centroid()
        edge_center3 = ee.Geometry.LineString([corner3.coordinates(), corner4.coordinates()]).centroid()
        edge_center4 = ee.Geometry.LineString([corner4.coordinates(), corner1.coordinates()]).centroid()

        # 创建包含所有点的MultiPoint几何对象
        region_coord = ee.Geometry.MultiPoint([
                        corner1.coordinates(), corner2.coordinates(), 
                        corner3.coordinates(), corner4.coordinates(),
                        edge_center1.coordinates(), edge_center2.coordinates(), 
                        edge_center3.coordinates(), edge_center4.coordinates(),
                        point.coordinates()])
    else:
        print('neighborhood_type should be "4" or "9"')

    # 使用MultiPoint几何对象作为geometry参数
    result = image.reduceRegion(
        reducer=ee.Reducer.toList(),  # 转换为列表
        geometry=region_coord,
        scale=reduceSacle,
        maxPixels=1e9)

    # 在结果中添加点坐标
    result = result.set('point_coordinates', point.coordinates())
    if Pointfilter:
        result = ee.Algorithms.If(ee.List(result.get('angle')).size().gte(int(neighborhood_type)), result, None)
    return result 

def weighted_avg_func(neighbor):
    # 使用ee.Number转换index，因为原始的index是ee.ComputedObject
    neighbors_info = ee.Dictionary(neighbor)

    # 将邻域点信息转换为ee.Array
    lon_coords = ee.Array(neighbors_info.get('longitude'))
    lat_coords = ee.Array(neighbors_info.get('latitude'))
    elevations = ee.Array(neighbors_info.get('elevation'))
    x_ = ee.Array(neighbors_info.get('x'))
    y_ = ee.Array(neighbors_info.get('y'))
    angles = ee.Array(neighbors_info.get('angle'))
    point_coords  = ee.List(neighbors_info.get('point_coordinates'))
    
    # 计算距离和权重
    point1_x = ee.Number(point_coords.get(0))
    point1_y = ee.Number(point_coords.get(1))
    distances = lon_coords.subtract(point1_x).pow(2).add(lat_coords.subtract(point1_y).pow(2)).sqrt()
    weights = distances.pow(-1)
    sum_weights = weights.reduce(ee.Reducer.sum(), [0])

    # 计算加权平均高程
    weighted_elevations = elevations.multiply(weights)
    weighted_avg_elevation = weighted_elevations.reduce(ee.Reducer.sum(), [0]).divide(sum_weights).get([0])

    # 计算平均角度
    weighted_angles = angles.multiply(weights)
    weighted_avg_angles = weighted_angles.reduce(ee.Reducer.sum(), [0]).divide(sum_weights).get([0])
    
    # 计算平均x
    weighted_X = x_.multiply(weights)
    weighted_avg_X = weighted_X.reduce(ee.Reducer.sum(), [0]).divide(sum_weights).get([0])
    
    # 计算平均y
    weighted_Y = y_.multiply(weights)
    weighted_avg_Y = weighted_Y.reduce(ee.Reducer.sum(), [0]).divide(sum_weights).get([0])

    return ee.Dictionary({'elevation': weighted_avg_elevation,'angle':weighted_avg_angles,
                          'x':weighted_avg_X,'y':weighted_avg_Y,'point_coordinates':neighbors_info.get('point_coordinates')})

def avg_func(neighbor):
    neighbors_info = ee.Dictionary(neighbor)
    elevations = ee.Array(neighbors_info.get('elevation'))
    angles = ee.Array(neighbors_info.get('angle'))
    x_ = ee.Array(neighbors_info.get('x'))
    y_ = ee.Array(neighbors_info.get('y'))
    return ee.Dictionary({'elevation':elevations.reduce(ee.Reducer.mean(),[0]).get([0]) ,
                           'angle':angles.reduce(ee.Reducer.mean(),[0]).get([0]),
                           'x':x_.reduce(ee.Reducer.mean(),[0]).get([0]),
                           'y':y_.reduce(ee.Reducer.mean(),[0]).get([0]),
                           'point_coordinates':neighbors_info.get('point_coordinates')})

def Volum9_func(neighbors_info):
    
    # 修改方程函数以匹配新的模型
    def equation(params, x, y):
        a, b, c, d, e, f = params
        return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

    # 修改残差函数，去除z作为变量
    def residuals(params, x, y, z_true):
        z_pred = equation(params, x, y)
        return z_pred - z_true
    
    # 根据求得的参数和给定点的经纬度计算高程
    def calculate_z(params, x, y):
        # 根据新模型参数更新，只有6个参数
        a, b, c, d, e, f = params
        # 更新方程以反映新的模型形式
        z_pred = a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
        return z_pred
    

    lon = np.array(neighbors_info['longitude'])
    lat = np.array(neighbors_info['latitude'])
    elevation = np.array(neighbors_info['elevation'])
    angles = np.array(neighbors_info['angle'])
    x_ = np.array(neighbors_info['x'])
    y_ = np.array(neighbors_info['y'])
    lon_pre, lat_pre = np.array(neighbors_info['point_coordinates'])

    # 初始参数猜测
    initial_guess = np.zeros(6)

    # 使用最小二乘法求得的参数,预测高程
    result0 = least_squares(residuals, initial_guess, args=(lon, lat, elevation))
    params0 = result0.x  
    z_pred = calculate_z(params0, lon_pre, lat_pre)

    # 使用最小二乘法求得的参数,预测angle
    result1 = least_squares(residuals, initial_guess, args=(lon, lat, angles))
    params1 = result1.x  
    angle_pred = calculate_z(params1, lon_pre, lat_pre)
    
    # 使用最小二乘法求得的参数,预测x
    result2 = least_squares(residuals, initial_guess, args=(lon, lat, x_))
    params2 = result2.x  
    x_pred = calculate_z(params2, lon_pre, lat_pre)
    
    # 使用最小二乘法求得的参数,预测y
    result3 = least_squares(residuals, initial_guess, args=(lon, lat, y_))
    params3 = result3.x  
    y_pred = calculate_z(params3, lon_pre, lat_pre)
    
    return {'elevation':z_pred,'angle':angle_pred,
            'x':x_pred,'y':y_pred,
            'point_coordinates':[lon_pre, lat_pre]}

def Flat4_func(neighbors_info):

    lon = neighbors_info['longitude']
    lat = neighbors_info['latitude']
    elevation = neighbors_info['elevation']
    angles = np.array(neighbors_info['angle'])
    x_ = np.array(neighbors_info['x'])
    y_ = np.array(neighbors_info['y'])
    lon_pred, lat_pred = np.array(neighbors_info['point_coordinates'])
    
    A = np.vstack([lon, lat, np.ones(len(lon))]).T  # 构建矩阵A

    b = elevation  # 高程向量b
    # 使用最小二乘法解方程Ax = b
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    # x包含拟合参数a, b, c（注意这里不再是1/c）
    a, b, c = x
    z_pred = a * lon_pred + b * lat_pred + c

    b = angles
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    a, b, c = x
    angle_pred = a * lon_pred + b * lat_pred + c
    
    b = x_
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    a, b, c = x
    x_pred = a * lon_pred + b * lat_pred + c
    
    b = y_
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    a, b, c = x
    y_pred = a * lon_pred + b * lat_pred + c

    return {'elevation':z_pred,'angle':angle_pred,'x':x_pred,'y':y_pred,'point_coordinates':[lon_pred, lat_pred]}

def Bilinear_interp_func(neighbors_info):
    lon = np.array(neighbors_info['longitude'])
    lat = np.array(neighbors_info['latitude'])
    elevation = np.array(neighbors_info['elevation'])
    angles = np.array(neighbors_info['angle'])
    x_ = np.array(neighbors_info['x'])
    y_ = np.array(neighbors_info['y'])
    lon_pred, lat_pred = np.array(neighbors_info['point_coordinates'])

    # 确定目标点的四个最近邻点
    distances = np.sqrt((lon - lon_pred)**2 + (lat - lat_pred)**2)
    idx = np.argsort(distances)[:4]
    lon = lon[idx]
    lat = lat[idx]
    elevation = elevation[idx]
    angles = angles[idx]
    x_ = x_[idx]
    y_ = y_[idx]

    # 计算双线性插值
    def bilinear_interp(values):
        # 根据经纬度构造插值矩阵
        matrix = np.array([
            [1, lon[0], lat[0], lon[0] * lat[0]],
            [1, lon[1], lat[1], lon[1] * lat[1]],
            [1, lon[2], lat[2], lon[2] * lat[2]],
            [1, lon[3], lat[3], lon[3] * lat[3]]
        ])
        # 解线性方程组以获得插值系数
        coeffs = np.linalg.solve(matrix, values)
        # 使用插值系数计算预测值
        return coeffs[0] + coeffs[1] * lon_pred + coeffs[2] * lat_pred + coeffs[3] * lon_pred * lat_pred

    z_pred = bilinear_interp(elevation)
    angle_pred = bilinear_interp(angles)
    x_pred = bilinear_interp(x_)
    y_pred = bilinear_interp(y_)

    return {'elevation': z_pred, 'angle': angle_pred, 'x': x_pred, 'y': y_pred, 'point_coordinates': [lon_pred, lat_pred]}

def apply_map_to_list(input_list, func): return input_list.map(func)

def apply_map_to_list_local(input_list, func): return map(func,input_list)

def Main_CalNeighbor(Templist,AOI,Prj_scale,Cal_image,Neighbors='4',Elvevation_model='weighted_avg_elevation'):
    Len_Templist =  Templist.size().getInfo()
    Neighbors = Neighbors # '9'
    Elvevation_model = Elvevation_model #'weighted_avg_elevation' , 'avg_elevation', 'Area_elavation','Volum_elavation','Bilinear_interp'

    All_PointLine = [] 
    pbar = range(Len_Templist)
    for i in pbar:
        points = Line2Points(Templist.get(i),region=AOI,scale=Prj_scale)  
        points = points.geometry().coordinates()
        All_PointLine.append(points)
        
    EE_PointLine = ee.List(All_PointLine)

    # 过滤点数不符合要求的辅助线
    EE_PointLine = EE_PointLine.map(partial(filter_Listlen,len=3)).removeAll([None])

    cal_neighbors = apply_map_to_list(EE_PointLine, 
                                    lambda x: ee.List(x).map(partial(get_neighborhood_info,image=Cal_image,
                                                                    scale = Prj_scale // 2 if Neighbors == '4' else Prj_scale,
                                                                    reduceSacle = Prj_scale,
                                                                    neighborhood_type=Neighbors,
                                                                    Pointfilter=False)).removeAll([None]))

    cal_neighbors = cal_neighbors.map(partial(filter_Listlen,len=3)).removeAll([None])

    if Elvevation_model == 'weighted_avg_elevation':
        Points_WithH_Angle = apply_map_to_list(
                                            cal_neighbors,      
                                            lambda x: ee.List(x).map(weighted_avg_func))
        Points_WithH_Angle = Points_WithH_Angle.getInfo()
    elif Elvevation_model == 'avg_elevation':
        Points_WithH_Angle = apply_map_to_list(
                                            cal_neighbors,
                                            lambda x: ee.List(x).map(avg_func))
        Points_WithH_Angle = Points_WithH_Angle.getInfo()
    elif (Elvevation_model == 'Area_elavation') or (Elvevation_model == 'Volum_elavation') or (Elvevation_model == 'Bilinear_interp'):
        neighbors_info = cal_neighbors.getInfo() # 非常大
        if Elvevation_model == 'Area_elavation':
            Points_WithH_Angle = list(apply_map_to_list_local(neighbors_info,lambda x: list(map(Flat4_func,x))))
        elif Elvevation_model == 'Volum_elavation':
            Points_WithH_Angle = list(apply_map_to_list_local(neighbors_info,lambda x: list(map(Volum9_func,x))))
        elif Elvevation_model == 'Bilinear_interp':
            Points_WithH_Angle = list(apply_map_to_list_local(neighbors_info,lambda x: list(map(Bilinear_interp_func,x))))
    return Points_WithH_Angle


from GEE_Tools import Select_imageNum
from PackageDeepLearn.utils import DataIOTrans

DomainDistorTest = ee.FeatureCollection('projects/ee-mrwurenzhe/assets/Test/DEM_ReConstruct')
Mountain = Select_imageNum(DomainDistorTest,0)
DEM = ee.Image("NASA/NASADEM_HGT/001").select('elevation')

# 设置参数
year = '2019'
START_DATE  = ee.Date(year + '-01-01')
END_DATE   = ee.Date(year + '-12-30')
TIME_LEN   = END_DATE.difference(START_DATE, 'days').abs()
MIDDLE_DATE = START_DATE.advance(TIME_LEN.divide(ee.Number(2)).int(),'days')
Origin_scale = 10
Prj_scale = 30
Savpath = DataIOTrans.make_dir(r'D:\test')
os.chdir(Savpath)

AOI = ee.Feature(Mountain).geometry()
s1_ascending, s1_descending = load_S1collection(AOI,START_DATE,END_DATE,MIDDLE_DATE,FilterSize=30)
Orbit = 'ASCENDING'
S1_image = s1_ascending
Projection = S1_image.select(0).projection()
Mask = S1_image.select(0).mask()
# 获取辅助线
azimuthEdge, rotationFromNorth, startpoint, endpoint, coordinates_dict  = S1Corrector.getS1Corners(S1_image, AOI, Orbit) 
Heading = azimuthEdge.get('azimuth')
s1_azimuth_across = ee.Number(Heading).subtract(90.0) # 距离向
Auxiliarylines = ee.Geometry.LineString([startpoint, endpoint])
BandTrans.delBands(S1_image, ['VH', 'VV'])

Cal_image = (Eq_pixels(BandTrans.delBands(S1_image, 'VV','VH').resample('bicubic')).rename('angle')
                    .addBands(ee.Image.pixelCoordinates(Projection))
                    .addBands(DEM.select('elevation'))
                    .addBands(ee.Image.pixelLonLat())
                    .updateMask(Mask)
                    .reproject(crs=Projection, scale=Prj_scale)
                    .clip(AOI))

Projection = Cal_image.select(0).projection()
Templist = S1_CalDistor.AuxiliaryLine2Point(s1_azimuth_across, coordinates_dict, Auxiliarylines,AOI, Prj_scale)
print('完成Templist')

Len_Templist =  Templist.size().getInfo()
Neighbors='4'
#'weighted_avg_elevation' , 'avg_elevation', 'Area_elavation','Volum_elavation','Bilinear_interp'
Elvevation_model='weighted_avg_elevation' 
All_PointLine = [] 

pbar = range(Len_Templist)
for i in pbar:
    points = Line2Points(Templist.get(i),region=AOI,scale=Prj_scale)  
    points = points.geometry().coordinates()
    All_PointLine.append(points)
EE_PointLine = ee.List(All_PointLine)

EE_PointLine = EE_PointLine.map(partial(filter_Listlen,len=3)).removeAll([None])
cal_neighbors = apply_map_to_list(EE_PointLine, 
                                lambda x: ee.List(x).map(partial(get_neighborhood_info,image=Cal_image,
                                                                scale = Prj_scale // 2 if Neighbors == '4' else Prj_scale,
                                                                reduceSacle = Prj_scale,
                                                                neighborhood_type=Neighbors,
                                                                Pointfilter=False)).removeAll([None]))
cal_neighbors = cal_neighbors.map(partial(filter_Listlen,len=3)).removeAll([None])
Points_WithH_Angle = apply_map_to_list(
                    cal_neighbors,      
                    lambda x: ee.List(x).map(weighted_avg_func))
Points_WithH_Angle = Points_WithH_Angle.getInfo()

from shapely.geometry import Point
import geopandas as gpd
points = [Point(item['point_coordinates']) for sublist in Points_WithH_Angle for item in sublist]

gdf = gpd.GeoDataFrame({
    'angle': [item['angle'] for sublist in Points_WithH_Angle for item in sublist],
    'elevation': [item['elevation'] for sublist in Points_WithH_Angle for item in sublist],
    'x': [item['x'] for sublist in Points_WithH_Angle for item in sublist],
    'y': [item['y'] for sublist in Points_WithH_Angle for item in sublist],
    'geometry': points
})
epsg_code = 4326
gdf.set_crs(epsg_code, inplace=True)

# 保存为Shapefile
gdf.to_file('{}.shp'.format(Elvevation_model))

# Points_WithH_Angle = Main_CalNeighbor(Templist,AOI,Prj_scale,Cal_image,Neighbors='4',Elvevation_model='weighted_avg_elevation')
# print('完成Points_WithH_Angle')