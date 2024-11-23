import geopandas as gpd
import shapely
from shapely.ops import unary_union,nearest_points
from shapely.geometry import LineString, Point,Polygon,MultiPoint
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d,RBFInterpolator
import plotly.graph_objects as go
import plotly.io as pio
import plotly.tools as tls
from pyproj import Transformer
import os

# Set the renderer
pio.renderers.default = "browser"

# Step 0: 载入分割面,排除边界外点，并采用对称3D插值
def read_split_poly(JsonPath,glacial_lakes,target_points=200):

    
    # 读取并提取数据
    line = gpd.read_file(JsonPath).geometry
    _, _, lake_level = zip(*line[0].coords)
    lake_level = np.mean(lake_level)
    x_obs, y_obs, z_obs = zip(*line[1].coords)

    # 创建一个GeoDataFrame来存储观测点
    obs_points = gpd.GeoDataFrame(geometry=[Point(xy) for xy in zip(x_obs, y_obs)], crs="EPSG:4326")
    
    lake_sort = float(os.path.basename(JsonPath).split('_')[0])
    glacial_lake = glacial_lakes[glacial_lakes["Sort"] == lake_sort]
    lake_geom_series = glacial_lake.geometry
    lake_geom = lake_geom_series.iloc[0]
    
    # 确保obs_points的CRS与lake_geom_series的CRS一致
    if obs_points.crs != lake_geom_series.crs:
        obs_points = obs_points.to_crs(lake_geom_series.crs)

    # 筛选出在湖泊内的观测点
    obs_points_inside = obs_points[obs_points.within(lake_geom_series.iloc[0])]

    # 如果没有点在湖泊内，返回空列表
    if obs_points_inside.empty:
        return [], [], [], np.nan

    # 提取筛选后的观测点数据
    x_obs = obs_points_inside.geometry.x.tolist()
    y_obs = obs_points_inside.geometry.y.tolist()
    z_obs = obs_points_inside.geometry.apply(lambda p: z_obs[obs_points_inside.geometry.tolist().index(p)]).tolist()

    # 将数据转换为numpy数组
    points = np.column_stack((x_obs, y_obs, z_obs))

    # 找到最低点的位置
    min_z_index = np.argmin(z_obs)

    # 分割点集
    left_points = points[:min_z_index+1]
    if len(left_points) !=  len(points):
        right_points = points[min_z_index:]
        left_interpolated = interpolate_points_3D(left_points, target_points=target_points//2)
        right_interpolated = interpolate_points_3D(right_points, target_points=target_points//2 + 1)
        
        # 对左右两段进行插值
        left_interpolated = interpolate_points_3D(left_points, target_points=target_points//2)
        right_interpolated = interpolate_points_3D(right_points, target_points=target_points//2 + 1)

        # 去除最低点重复的部分
        right_interpolated = right_interpolated[1:]
        interpolated_points = np.concatenate((left_interpolated, right_interpolated), axis=0)
        
        # 提取插值后的x、y和z
        x_obs = interpolated_points[:, 0]
        y_obs = interpolated_points[:, 1]
        z_obs = interpolated_points[:, 2]
        
    else:
        left_interpolated = interpolate_points_3D(left_points, target_points=target_points)
        left_interpolated = interpolate_points_3D(left_points, target_points=target_points)
        x_obs = left_interpolated[:, 0]
        y_obs = left_interpolated[:, 1]
        z_obs = left_interpolated[:, 2]
          
    # 镜像处理z值以保持对称性
    z_obs = np.where(np.arange(len(z_obs)) < len(z_obs)//2, z_obs, z_obs[::-1])
    
    return x_obs, y_obs, z_obs, lake_level,lake_geom,lake_sort

# Step 1: 等距离插值
def interpolate_points(points, target_points=500):
    if len(points) < target_points:
        distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)
        total_length = distances[-1]
        new_points = np.linspace(0, total_length, target_points)
        f_x = interp1d(distances, points[:, 0], kind='linear')
        f_y = interp1d(distances, points[:, 1], kind='linear')
        return np.column_stack((f_x(new_points), f_y(new_points)))
    else:
        return points
def interpolate_points_3D(points, target_points=500):
    if len(points) < target_points:
        # 计算每个点之间的距离
        distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)
        total_length = distances[-1]
        
        # 生成新的等距离点
        new_points = np.linspace(0, total_length, target_points)
        
        # 插值x、y、z
        f_x = interp1d(distances, points[:, 0], kind='linear')
        f_y = interp1d(distances, points[:, 1], kind='linear')
        f_z = interp1d(distances, points[:, 2], kind='linear')
        
        return np.column_stack((f_x(new_points), f_y(new_points), f_z(new_points)))
    else:
        return points

# Step 2: 找到稳定的中心点
def find_stable_centroid(lake_geom, buffer_distance=30, iterations=10, tolerance=10, origin_crs='EPSG:4326', proj_crs='EPSG:8859'):
    geom_series = gpd.GeoSeries([lake_geom], crs=origin_crs)
    geom_utm = geom_series.to_crs(proj_crs)
    current_geom = geom_utm.iloc[0]
    centroids = []
    last_centroid = None
    for _ in range(iterations):
        buffered_geom = current_geom.buffer(-buffer_distance)
        if buffered_geom.is_empty:
            break
        if isinstance(buffered_geom, shapely.geometry.MultiPolygon):
            for poly in buffered_geom.geoms:
                centroid = poly.centroid
                if poly.contains(centroid):
                    centroids.append(centroid)
            break
        else:
            centroid = buffered_geom.centroid
            if buffered_geom.contains(centroid):
                centroids = [centroid]
        if last_centroid and centroids[-1].distance(last_centroid) < tolerance:
            break
        if len(centroids) !=0:
            last_centroid = centroids[-1]
        current_geom = buffered_geom
    centroids_wgs84 = gpd.GeoSeries(centroids, crs=proj_crs).to_crs(origin_crs)
    return list(centroids_wgs84)
def find_representative_point(lake_geom):
    return [lake_geom.representative_point()]
    
# Step 3: 找到与中心点的延长线与多边形边界的交点与观测点距离多边形最近点
def find_most_far_point_pair(intersection):
    points = list(intersection.geoms)
    if len(points) > 2:
        # 计算所有交点之间的距离
        max_distance = 0; i_ = 0; j_ = 0;
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if points[i].distance(points[j]) > max_distance:
                    max_distance = points[i].distance(points[j])
                    i_ = i; j_ = j
    elif len(points) == 2:
        i_ = 0; j_ = 1
    return points[i_] ,points[j_]
def find_intersection_with_center(x_center, y_center, x_obs, y_obs, x_b, y_b, extension_factor=1000):
    intersections = []
    nears = []
    boundary_line = LineString(np.column_stack((x_b, y_b)))
    for x, y in zip(x_obs, y_obs):
        vector = np.array([x - x_center, y - y_center])
        unit_vector = vector / np.linalg.norm(vector)
        extended_start = np.array([x_center, y_center]) - extension_factor * unit_vector
        extended_end = np.array([x_center, y_center]) + extension_factor * unit_vector
        line = LineString([tuple(extended_start), tuple(extended_end)])
        intersection = line.intersection(boundary_line)
        
        # 使用geom_type替代type
        if intersection.geom_type == 'MultiPoint':
            intersections.append((find_most_far_point_pair(intersection)))
        #     else:
        #         intersections.append(points[0])
        # else:
        #     intersections.append(intersection)
        
        # 计算最近点
        nearest_point, _ = nearest_points(boundary_line, Point(x, y))
        nears.append(nearest_point)
    return intersections, nears

# Step 4: 计算观测点到交点的距离，并选择最近交点
def calculate_distances_to_intersections(df, x_center, y_center, projection_crs="EPSG:8859"):
    # 投影中心点
    center_point = gpd.GeoSeries([Point(x_center, y_center)], crs="EPSG:4326").to_crs(projection_crs)
    for i in range(len(df)):
        obs_geo = df.loc[i, 'obs_geo']
        # 投影交点
        intersect_points = [Point(df.loc[i, 'intersect_points_0']), Point(df.loc[i, 'intersect_points_1'])]
        intersect_points_projected = gpd.GeoSeries(intersect_points, crs="EPSG:4326").to_crs(projection_crs)
        # 计算观测点到交点和最近点的距离
        distances = [obs_geo.distance(point) for point in intersect_points_projected]
        df.loc[i,'Distance_to_Intersection_min'] = np.min(distances)
        # 找到距离最近的交点
        min_distance_index = np.argmin(distances)
        nearest_intersect_point = intersect_points_projected.iloc[min_distance_index]
        df.loc[i, 'nearest_intersect_point'] = intersect_points[min_distance_index]
        # 计算最近交点到中心点的距离
        df.loc[i, 'Distance_to_Center'] = nearest_intersect_point.distance(center_point.iloc[0])

# Step 5: 计算中心点到边界收缩比率，并计算不同高度的边界
def interpolate_circular(data):
    # 找到非NaN值的索引
    non_nan_indices = np.where(~np.isnan(data))[0]
    
    if len(non_nan_indices) < 2:
        # 如果没有足够的已知值，无法进行插值，直接返回原数组
        return data
    
    # 初始化插值结果数组
    interpolated_data = np.copy(data)
    
    # 对每个已知值对进行插值
    for i in range(len(non_nan_indices)):
        start_idx = non_nan_indices[i]
        if i == len(non_nan_indices) - 1:
            # 处理最后一个已知值到第一个已知值的环形连接
            end_idx = non_nan_indices[0] + len(data)
        else:
            end_idx = non_nan_indices[i+1]
        
        # 截取当前段的索引和数据
        segment_indices = np.arange(start_idx, end_idx+1) % len(data)
        segment_data = data[segment_indices]
        
        # 找到当前段内的非NaN值
        segment_non_nan = np.where(~np.isnan(segment_data))[0]
        
        if len(segment_non_nan) >= 2:
            # 使用线性插值
            start_value = segment_data[segment_non_nan[0]]
            end_value = segment_data[segment_non_nan[-1]]
            interpolated_data[segment_indices] = np.linspace(start_value, end_value, len(segment_data))
        else:
            # 如果只有一个已知值，计算线性变化并填充
            known_value = segment_data[segment_non_nan[0]]
            prev_idx = (start_idx - 1) % len(data)
            next_idx = (end_idx - 1) % len(data)
            prev_value = data[prev_idx] if not np.isnan(data[prev_idx]) else known_value
            next_value = data[next_idx] if not np.isnan(data[next_idx]) else known_value
            interpolated_data[segment_indices] = np.linspace(prev_value, next_value, len(segment_data))

    return interpolated_data
def find_indices_in_list(points, all_points):
    indices = []
    used_indices = set()
    for point in points:
        distances = [point.distance(p) for p in all_points]
        sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
        
        # 找到最接近的索引，如果已经使用过则找第二接近的
        for idx in sorted_indices:
            if idx not in used_indices:
                indices.append(idx)
                used_indices.add(idx)
                break
            elif idx == sorted_indices[-1]:  # 如果所有索引都已使用，返回None或处理错误
                print(f"Warning: No unique index available for point {point}")
                indices.append(None)
                break
    return indices
def calculate_shrink_boundary(df,all_boundary_points):
    unique_heights = df['z_obs'].unique()
    height_shrink = {}
    for height in unique_heights:
        height_points = df[df['z_obs'] == height]
        origin_bound_points = height_points['nearest_intersect_point']
        # shrink_bound_points = height_points.apply(lambda row: Point(row['x_obs'], row['y_obs']), axis=1)
        
        # 找到origin_bound_points在all_boundary_points中的序号
        indices = find_indices_in_list(origin_bound_points, all_boundary_points)
        
        # 计算每个点对的收缩率
        shrink_ratios = height_points['Distance_to_Intersection_min'] / height_points['Distance_to_Center']
        
        template_array = np.full(len(all_boundary_points), np.nan)
        # 将收缩率填充到对应的位置
        for i, idx in enumerate(indices):
            template_array[idx] = shrink_ratios.iloc[i]
        interpolated_array = interpolate_circular(template_array)
        height_shrink[height] = {
            'ratios': shrink_ratios.tolist(),
            'indices': indices,
            'interpolated_array':interpolated_array}
    return height_shrink   
def calculate_shrunk_boundary_points(all_boundary_points, height_shrink, x_center, y_center):
    shrunk_points = {}

    for height, data in height_shrink.items():
        interpolated_ratios = data['interpolated_array']
        shrunk_points[height] = []
        
        for i, boundary_point in enumerate(all_boundary_points):
            original_x, original_y = boundary_point.x, boundary_point.y
            shrink_ratio = interpolated_ratios[i]
            
            delta_x = original_x - x_center
            delta_y = original_y - y_center
            
            shrunk_x = x_center + delta_x * (1-shrink_ratio)
            shrunk_y = y_center + delta_y * (1-shrink_ratio)
            
            shrunk_points[height].append(Point(shrunk_x, shrunk_y))

    return shrunk_points

# Step 6: 删除不合理的边界，下层大于上层
def remove_excessive_or_intersecting_boundaries(shrunk_points, lake_geom=None):
    # 按高度从高到低排序
    sorted_heights = sorted(shrunk_points.keys(), reverse=True)
    
    # 创建一个新的字典来存储过滤后的边界
    filtered_boundaries = {}
    
    # 初始的边界集合
    all_previous_boundaries = [lake_geom]

    for i in range(len(sorted_heights)):
        current_height = sorted_heights[i]
        current_boundary = [point.coords[0] for point in shrunk_points[current_height]]
        current_polygon = Polygon(current_boundary)
        
        # 初始化标志，表示是否应该保留当前边界
        keep_boundary = True
        
        # 检查当前边界是否与任何已保留的边界相交或面积是否大于等于其中任何一个
        for prev_boundary in all_previous_boundaries:
            if current_polygon.exterior.intersects(prev_boundary.exterior) or current_polygon.area >= prev_boundary.area:
                keep_boundary = False
                break
        
        if keep_boundary:
            # 如果不相交且面积小于所有上层边界，则添加当前边界
            filtered_boundaries[current_height] = [Point(point) for point in current_boundary]
            # 将当前边界加入到已保留的边界列表中
            all_previous_boundaries.append(current_polygon)
    
    return filtered_boundaries

# Step 7: 采用二次样条函数，计算过最深点的交线与构造边界的交点（最深点可不采用，导入会产生误差）
# 获取湖泊最深处的高程，并将构造的二次样条函数值用于湖泊底部生成
def peak_valley_count(y_interp,z_interp,mask):
    try:
        dz = np.gradient(z_interp[mask], y_interp[mask])
        zero_crossings = np.where(np.diff(np.sign(dz)))[0]
        peaks = []
        valleys = []
        for i in zero_crossings:
            if dz[i] > 0 and dz[i+1] < 0:
                peaks.append(y_interp[i])
            elif dz[i] < 0 and dz[i+1] > 0:
                valleys.append(y_interp[i])
        return peaks, valleys
    except:
        return [1,1,1,1], [1,1,1,1]
def calculate_center_depth(x_center, y_center, z_obs,
                           shrunk_points, lake_geom, lake_level,plot=True):
    
    closest_point_index = np.argmin(z_obs)
    intersection_line = LineString(intersections[closest_point_index])
    sorted_heights = sorted(shrunk_points.keys(), reverse=True)
    # x_in, y_in,z_in =  x_obs[closest_point_index], y_obs[closest_point_index],z_obs[closest_point_index]
    
    boundarys = [lake_geom.exterior]; heights = [lake_level]
    for current_height in sorted_heights:
        boundarys.append(Polygon([point.coords[0] for point in shrunk_points[current_height]]).exterior)
        heights.append(current_height)
    if len(boundarys)> 2:
        del boundarys[0] ; del heights[0]
    
    intersection_points = intersection_line.intersection(boundarys)
    if len(boundarys)<=2:
        intersection_points[0] = MultiPoint(intersections[closest_point_index])
    x = np.zeros(len(intersection_points) * 2) ; y = np.zeros(len(intersection_points) * 2)
    for j,multipoint in enumerate(intersection_points):
        for i,point in enumerate(multipoint.geoms):
            if i == 0:
                x[j] = point.x; y[j] = point.y
            elif i == 1:
                x[-(j+1)] = point.x; y[-(j+1)] = point.y

    z = np.array(heights + heights[::-1])
    
    # 创建样条插值函数
    f = interp1d(y, z, kind='quadratic', fill_value='extrapolate') #  一次样条插值'linear'，二次样条插值'quadratic' , 三次'cubic'

    # 生成一个y范围来绘制插值线
    xy_interp = interpolate_points(np.array([[i,j] for i,j in zip(x,y)]), target_points=100)
    
    x_interp = np.array([each[0] for each in xy_interp])
    y_interp = np.array([each[1] for each in xy_interp])
    z_interp = f(y_interp)
    z_center = f(y_center)
    
    sorted_indices = np.argsort(y_interp)
    y_interp = y_interp[sorted_indices]
    x_interp = x_interp[sorted_indices]
    z_interp = z_interp[sorted_indices]
    
    # 判断波峰和波谷
    min_y_values = y[np.argsort(z)[:2]]
    mask = (y_interp >= min(min_y_values)) & (y_interp <= max(min_y_values))
    peaks, valleys = peak_valley_count(y_interp,z_interp,mask)

    # 检查是否有多个波峰或波谷
    if len(peaks) + len(valleys) > 1:
        print("存在多个波峰或波谷,重新插值")
        indice = [0, len(y)//2-1,len(y)//2, len(y)-1]
        f = interp1d(y[indice], z[indice], kind='quadratic', fill_value='extrapolate')
        z_interp = f(y_interp)
        z_center = f(y_center)
    else:
        print("只有一个波峰或波谷")
        
    # 此时仅返回以下值，因为不与已知点相交
    x_interp=x_interp[mask]
    y_interp=y_interp[mask]
    z_interp=z_interp[mask]
    
    if plot == True:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.scatter(y, z, label='Original data')
        plt.plot(y_interp, z_interp, color='red', label='样条插值')
        plt.scatter(y_center, z_center, color='green', s=100, label='center point')  # 标注出查找的点
        # plt.scatter(y_in, z_in, color='blue', s=100, label='insert point')
        # 设置标签和图例
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.title('Y-Z 样条插值')
        plt.legend()

        # 显示图形
        plt.show()
    
    return z_center,x_interp,y_interp,z_interp

def calculate_others(x_center, y_center, z_center, z_obs, shrunk_points,lake_geom, lake_level):
    closest_point_index = np.argmin(z_obs)
    del intersections[closest_point_index]  
    intersection_lines = [LineString(each) for each in intersections]
    sorted_heights = sorted(shrunk_points.keys(), reverse=True)
    
    boundarys = [lake_geom.exterior]; heights = [lake_level]
    for current_height in sorted_heights:
        boundarys.append(Polygon([point.coords[0] for point in shrunk_points[current_height]]).exterior)
        heights.append(current_height)
    if len(boundarys)> 2:
        del boundarys[0] ; del heights[0]    
        
    x_interps = []; y_interps = []; z_interps=[]; mins = []
    for point_index,intersection_line in enumerate(intersection_lines):
        intersection_points = intersection_line.intersection(boundarys)
        if len(boundarys)<=2:
            intersection_points[0] = MultiPoint(intersections[point_index])
        x = np.zeros(len(intersection_points) * 2) ; y = np.zeros(len(intersection_points) * 2)
        for j,multipoint in enumerate(intersection_points):
            point_pairs = find_most_far_point_pair(multipoint)
            for i,point in enumerate(point_pairs):
                if i == 0:
                    x[j] = point.x; y[j] = point.y
                elif i == 1:
                    x[-(j+1)] = point.x; y[-(j+1)] = point.y

        z = np.array(heights + heights[::-1])    
                
        mid_index = len(y) // 2
        new_x = np.zeros(len(x) + 2)
        new_x[:mid_index] = x[:mid_index]
        new_x[mid_index:mid_index+2] = [x_center,x_center+0.00001]
        new_x[mid_index+2:] = x[mid_index:]
        
        new_y = np.zeros(len(y) + 2)
        new_y[:mid_index] = y[:mid_index]
        new_y[mid_index:mid_index+2] = [y_center,y_center+0.00001]
        new_y[mid_index+2:] = y[mid_index:]
        
        new_z = np.zeros(len(z) + 2)
        new_z[:mid_index] = z[:mid_index]
        new_z[mid_index:mid_index+2] = [z_center,z_center+0.00001]
        new_z[mid_index+2:] = z[mid_index:]
        
        f = interp1d(new_y, new_z, kind='quadratic', fill_value='extrapolate') #  一次样条插值'linear'，二次样条插值'quadratic' , 三次'cubic'

        # 生成一个y范围来绘制插值线
        xy_interp = interpolate_points(np.array([[i,j] for i,j in zip(x,y)]), target_points=100)
        x_interp = np.array([each[0] for each in xy_interp])
        y_interp = np.array([each[1] for each in xy_interp])
        z_interp = f(y_interp)
        sorted_indices = np.argsort(y_interp)
        y_interp = y_interp[sorted_indices]
        x_interp = x_interp[sorted_indices]
        z_interp = z_interp[sorted_indices]
        
        # 判断波峰和波谷
        min_y_values = y[np.argsort(z)[:2]]
        mask = (y_interp >= min(min_y_values)) & (y_interp <= max(min_y_values))
        peaks, valleys = peak_valley_count(y_interp,z_interp,mask)

        # 检查是否有多个波峰或波谷
        if len(peaks) + len(valleys) > 1:
            print("存在多个波峰或波谷,重新插值")
            indice = [0,len(new_y)//2-2, len(new_y)//2-1,len(new_y)//2,len(new_y)//2+1, len(new_y)-1]
            f = interp1d(new_y[indice], new_z[indice], kind='quadratic', fill_value='extrapolate')
            z_interp = f(y_interp)
        else:
            print('单个波峰波谷')
            
        # 此时仅返回以下值，因为不与已知点相交
        x_interp=x_interp[mask]
        y_interp=y_interp[mask]
        z_interp=z_interp[mask]
        
        if len(z_interp) == 0:
            tolerate = False
        else:
            mins.append(np.min(z_interp))
            tolerate = np.abs((z_center - np.min(z_interp)))<=np.abs(lake_level-z_center) * 0.01
            
        # 部分可能值很少，可以用x进行插值
        if len(z_interp)>30 and tolerate:
            x_interps.append(x_interp)
            y_interps.append(y_interp)
            z_interps.append(z_interp)
        else:
            f = interp1d(new_x, new_z, kind='quadratic') 
            z_interp = f(x_interp)
            if len(z_interp) == 0:
                tolerate = False
            else:
                tolerate = np.abs((z_center - np.min(z_interp)))<=np.abs(lake_level-z_center) * 0.01
            if len(z_interp)>15 and tolerate:
                x_interps.append(x_interp)
                y_interps.append(y_interp)
                z_interps.append(z_interp)
                
    return ([item for sublist in x_interps for item in sublist],
            [item for sublist in y_interps for item in sublist],
            [item for sublist in z_interps for item in sublist])
        
# Step 8: 采用RBF插值器[https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html]
# 还原湖泊地形，并计算湖泊体积
def RBF_interpolation(shrunk_points,x_b,y_b,z_b,x_interp,y_interp,z_interp,
                      x_center,y_center,z_center):
    # 准备数据
    x_all = []
    y_all = []
    z_all = []
    for height, points in shrunk_points.items():
        for point in points:
            x_all.append(point.x)
            y_all.append(point.y)
            z_all.append(height)  # 这里高度即是深度
            
    x_all = np.concatenate([x_b,  x_all,x_interp,[x_center]]) 
    y_all = np.concatenate([y_b,  y_all,y_interp,[y_center]])
    z_all = np.concatenate([z_b,  z_all,z_interp,[z_center]])

    # 移除重复点
    unique_points, indices = np.unique(np.column_stack((x_all, y_all)), axis=0, return_index=True)
    x_all, y_all, z_all = x_all[indices], y_all[indices], z_all[indices]

    # Create a meshgrid for interpolation
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    xi = np.linspace(x_min, x_max, 100)
    yi = np.linspace(y_min, y_max, 100)
    xi, yi = np.meshgrid(xi, yi)

    # 使用RBFInterpolator进行插值
    rbfi = RBFInterpolator(np.column_stack((x_all, y_all)), z_all, kernel='linear', degree=0)
    zi = rbfi(np.column_stack((xi.ravel(), yi.ravel()))).reshape(xi.shape)
    zi[zi>lake_level] = lake_level
    
    return xi,yi,zi
def calculate_lake_volume(xi, yi, zi,lake_geom,lake_level,crs="EPSG:4326",projection_crs="EPSG:8859"):
    # 定义坐标转换器
    transformer = Transformer.from_crs(crs, projection_crs, always_xy=True)

    xi_flat = xi.ravel()
    yi_flat = yi.ravel()

    x_transformed, y_transformed = transformer.transform(xi_flat, yi_flat)

    xi_transformed = x_transformed.reshape(xi.shape)
    yi_transformed = y_transformed.reshape(yi.shape)

    lake_geom_proj = gpd.GeoSeries([lake_geom], crs=crs).to_crs(projection_crs).iloc[0]
    
    # 计算湖泊最小包络矩形"EPSG:8859"
    transformer = Transformer.from_crs(crs, projection_crs, always_xy=True)
    lake_geom_proj = gpd.GeoSeries([lake_geom], crs=crs).to_crs(projection_crs).iloc[0]
    # Minimum Rotated Rectangle
    min_rect = lake_geom_proj.minimum_rotated_rectangle
    boundary_points = list(min_rect.exterior.coords)
    lake_width = LineString([boundary_points[0], boundary_points[1]]).length
    lake_length = LineString([boundary_points[1], boundary_points[2]]).length
    if lake_width < lake_length:
        lake_width, lake_length = lake_length, lake_width
    print(f'最小包络矩形的长: {lake_length:.2f} 米')
    print(f'最小包络矩形的宽: {lake_width:.2f} 米')

    x_spacing = xi_transformed[0, 1] - xi_transformed[0, 0]
    y_spacing = yi_transformed[1, 0] - yi_transformed[0, 0]

    volume = 0 ; lake_deeps = []
    for i in range(zi.shape[0] - 1):
        for j in range(zi.shape[1] - 1):
            x_center = (xi_transformed[i, j] + xi_transformed[i+1, j+1]) / 2
            y_center = (yi_transformed[i, j] + yi_transformed[i+1, j+1]) / 2
            
            # 检查中心点是否在湖泊内
            if lake_geom_proj.contains(Point(x_center, y_center)):
                # 计算每个网格单元的高度（深度）
                height = (zi[i, j] + zi[i+1, j] + zi[i, j+1] + zi[i+1, j+1]) / 4
                cell_deep = lake_level - height
                cell_volume = cell_deep * x_spacing * y_spacing
                volume += cell_volume
                lake_deeps.append(cell_deep)

    print('lake volume = {:.2f} 10^6 m³, max deep = {:.2f} m, avg deep= {:.2f} m'
        .format(volume / (1000*1000), np.max(lake_deeps), np.mean(lake_deeps)))
    return volume / (1000*1000), np.max(lake_deeps),np.mean(lake_deeps),lake_width,lake_length

# Step 9: 绘图
def plot_lake_surface(xi, yi, zi, lake_level, x_center, y_center,z_center,x_interp,y_interp,z_interp,
                      x_b=None, y_b=None, z_b=None, 
                      x_obs=None, y_obs=None, z_obs=None, shrunk_points=None, df=None, 
                      add_boundary_points=True, add_spline_interp = True,
                      add_center_point=True, 
                      add_center_with_depth=True,add_observation_points=True, 
                      add_boundary_lines=True, add_intersection_lines=True,show=True):
    """
    绘制湖泊底部曲面及其相关点和线段。

    参数:
    - xi, yi, zi: 插值后的网格数据
    - lake_level: 湖泊水位
    - x_center, y_center: 湖泊中心点的坐标
    - x_interp,y_interp,z_interp 二次样条插值点
    - x_b, y_b, z_b: 湖泊边界点的坐标
    - x_obs, y_obs, z_obs: 观测点的坐标
    - shrunk_points: 一个字典，包含不同高度的边界点
    - df: 包含交点的DataFrame
    - add_boundary_points: 是否添加湖泊边界点
    - add_spline_interp: 是否添加二次样条插值点
    - add_center_point: 是否添加湖泊中心点
    - add_center_with_depth: 添加湖泊中心点(带高程)
    - add_observation_points: 是否添加观测点
    - add_boundary_lines: 是否添加边界点的水平线
    - add_intersection_lines: 是否添加交点线段
    """
    # 创建 Plotly 图形
    fig = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi)])

    # Customize the layout
    fig.update_layout(
        title='Interpolated Lake Depth Surface (RBF)',
        autosize=False,
        width=800,
        height=800,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Depth',
            aspectratio=dict(x=1, y=1, z=0.7),
            aspectmode='manual'
        ),
        scene_camera=dict(
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
    )

    if add_boundary_points and x_b is not None and y_b is not None and z_b is not None:
        fig.add_trace(
            go.Scatter3d(
                x=x_b,
                y=y_b,
                z=z_b,
                mode="markers",
                marker=dict(size=5, color="blue"),
                name="湖泊边界点"
            )
        )

    if add_spline_interp :
        fig.add_trace(
            go.Scatter3d(
                x=x_interp, 
                y=y_interp, 
                z=z_interp, 
                mode='markers',
                marker=dict(size=3, color="yellow"),
                name="过中心采样点"
            )
        )
    
    if add_center_with_depth:
        fig.add_trace(
            go.Scatter3d(
                x=[x_center], 
                y=[y_center], 
                z=[z_center], 
                mode='markers',
                marker=dict(size=10, color="red"),
                name="中心点"
            )
        )
        
    if add_center_point:
        fig.add_trace(
            go.Scatter3d(
                x=[x_center],
                y=[y_center],
                z=[lake_level],
                mode="markers",
                marker=dict(size=5, color="yellow"),
                name="湖泊中心点"
            )
        )

    if add_observation_points and x_obs is not None and y_obs is not None:
        fig.add_trace(
            go.Scatter3d(
                x=x_obs,
                y=y_obs,
                z=z_obs,
                mode="markers",
                marker=dict(size=5, color="red"),
                name="观测点"
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=x_obs,
                y=y_obs,
                z=(np.ones_like(x_obs) * lake_level),
                mode="markers",
                marker=dict(size=5, color="red", opacity=0.5),
                name="观测点_flat"
            )
        )

    if add_boundary_lines and shrunk_points:
        heights = list(shrunk_points.keys())
        for height in heights:
            points = shrunk_points[height]
            x_coords = [p.x for p in points if p is not None]
            y_coords = [p.y for p in points if p is not None]
            fig.add_trace(go.Scatter3d(
                x=x_coords, 
                y=y_coords, 
                z=[height] * len(x_coords),
                mode='markers',
                name=f'Boundary Points at Height {height}',
                marker=dict(size=5, color='blue', opacity=0.5)
            ))

    if add_intersection_lines and df is not None:
        for i in range(len(df)):
            x_line = [df['intersect_points_0'][i][0], df['intersect_points_1'][i][0]]
            y_line = [df['intersect_points_0'][i][1], df['intersect_points_1'][i][1]]
            z_line = [lake_level, lake_level]

            fig.add_trace(
                go.Scatter3d(
                    x=x_line,
                    y=y_line,
                    z=z_line,
                    mode="lines",
                    line=dict(color="green", width=2),
                    name=f"交点线段 {i}"
                )
            )

    # 设置布局
    fig.update_layout(
        title="湖泊底部曲面拟合",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        autosize=True,
    )
    if show:
        fig.show()
    return fig

def merge_polygons(geom, expand_distance=0.0001, erode_distance=0.0001):
    def dilate(geom, distance):
        return geom.buffer(distance)

    def erode(geom, distance):
        return geom.buffer(-distance)
    dilated = dilate(geom, expand_distance)
    eroded = erode(dilated, erode_distance)
    return unary_union(eroded)


# 分割面观测点，如果采用多根线，可能还需要考虑高程水平一致问题，可以考虑用interp1d线性插值，重新计算高度值以统一
from PackageDeepLearn.utils import file_search_wash as fsw
JSONPATHS = fsw.search_files(r'G:\SETP_ICESat-2\lake_volumn_by_ICEsat-2',endwith='.geojson')
glacial_lakes = gpd.read_file(r"D:\BaiduSyncdisk\02_论文相关\在写\SAM冰湖\数据\2023_05_31_to_2023_09_15_样本修正_SpatialJoin.shp")

for ex_num,JsonPath in enumerate(JSONPATHS):

    if ex_num < 0:
        continue

    # JsonPath = JSONPATHS[172]

    # -----------------------------载入观测点与湖泊范围
    x_obs, y_obs, z_obs, lake_level,lake_geom,lake_sort = read_split_poly(JsonPath,glacial_lakes,target_points=50)

    # -----------------------------读取湖泊边界点，并计算湖泊中心
    all_boundary_points = []
    if isinstance(lake_geom, shapely.geometry.MultiPolygon):
        # 对于每个多边形，提取其边界点
        merged_poly = merge_polygons(lake_geom, expand_distance=0.0003, erode_distance=0.0003)
        if isinstance(merged_poly, shapely.geometry.MultiPolygon):
            merged_poly = max(merged_poly.geoms, key=lambda x: x.area)
        lake_geom = merged_poly
    if isinstance(lake_geom, shapely.geometry.Polygon):
        boundary = lake_geom.exterior
        boundary_points = np.array([list(point) for point in boundary.coords])
        all_boundary_points.append(boundary_points)
    else:
        raise TypeError("Geometry type not supported for boundary extraction.")
    #湖泊边界， 只考虑polygen而不考虑mutipolygen
    all_boundary_points = [interpolate_points(poins,target_points=200) for poins in all_boundary_points]
    x_b = np.array([point[0] for point in all_boundary_points[0]])
    y_b = np.array([point[1] for point in all_boundary_points[0]])
    z_b = np.ones_like(x_b)*lake_level
    all_boundary_points = [Point(point)for point in all_boundary_points[0]]
    # stable_centroids可能有多个
    stable_centroids = find_stable_centroid(lake_geom, buffer_distance=15, iterations=10, tolerance=10) 

    if len(stable_centroids) == 1:
        x_center, y_center = stable_centroids[0].x, stable_centroids[0].y
        intersections,nears = find_intersection_with_center(x_center, y_center, x_obs, y_obs, x_b, y_b)
        intersect_points = [np.array([point.x, point.y]) for intersection in intersections for point in intersection]

        # 确保intersect_points的长度是偶数
        assert len(intersect_points) % 2 == 0

        points_on_section = gpd.GeoDataFrame(geometry=[Point(xy) for xy in zip(x_obs, y_obs)], crs="EPSG:4326").to_crs(epsg=8859)
        df = pd.DataFrame({
            'x_obs': x_obs,
            'y_obs': y_obs,
            'z_obs': z_obs,
            'obs_geo': points_on_section.geometry
        })
        df['intersect_points_0'] = intersect_points[::2]
        df['intersect_points_1'] = intersect_points[1::2]
        df['intersect_points_near'] = nears
        df = df.sort_values(by=['z_obs'], ascending=[False])
        calculate_distances_to_intersections(df, x_center, y_center,  projection_crs="EPSG:8859")
        height_shrink = calculate_shrink_boundary(df,all_boundary_points)
        shrunk_points = calculate_shrunk_boundary_points(all_boundary_points, height_shrink, x_center, y_center)
        
        shrunk_points = remove_excessive_or_intersecting_boundaries(shrunk_points,lake_geom)
        if len(shrunk_points) == 0:
            print("No shrunk points found.")
            continue
        
        z_center,x_interp,y_interp,z_interp= calculate_center_depth(x_center, y_center, z_obs,
                                                                    shrunk_points, lake_geom, lake_level,plot=False)
        x_interps,y_interps,z_interps = calculate_others(x_center, y_center, z_center, z_obs, shrunk_points,lake_geom, lake_level)

        x_interp = np.concatenate((x_interp, x_interps))
        y_interp = np.concatenate((y_interp, y_interps))
        z_interp = np.concatenate((z_interp, z_interps))
        
        xi,yi,zi = RBF_interpolation(shrunk_points,x_b,y_b,z_b,x_interp,y_interp,z_interp,x_center,y_center,z_center)
        lake_volume, lake_maxdeep, lake_meandeep,lake_width,lake_length = calculate_lake_volume(xi, yi, zi,lake_geom,lake_level,
                                                                        crs="EPSG:4326",projection_crs="EPSG:8859")
        
        
        fig = plot_lake_surface(xi, yi, zi, lake_level, x_center, y_center,z_center,x_interp,y_interp,z_interp,
                    x_b=x_b, y_b=y_b, z_b=z_b, 
                    x_obs=x_obs, y_obs=y_obs, z_obs=z_obs,
                    shrunk_points=shrunk_points, df=df,
                    add_boundary_points=True, add_spline_interp = True,
                    add_center_point=True, 
                    add_center_with_depth=True,add_observation_points=True, 
                    add_boundary_lines=True, add_intersection_lines=True,show=False)

        
        excel_filename = f"lake_data.csv"
        os.chdir(r'G:\SETP_ICESat-2\lake_volumn_by_ICEsat-2\plot_and_xlsx_record')
        # 记录湖泊数据
        lake_info = pd.DataFrame({
            'lake_sort': [lake_sort],
            'lake_level': [lake_level],
            'stable_centroids': [str(stable_centroids[0].coords[0])],  # 将Point对象转换为字符串
            'lake_volume': [lake_volume],
            'lake_maxdeep': [lake_maxdeep],
            'lake_meandeep': [lake_meandeep],
            'lake_width': [lake_width],
            'lake_length': [lake_length]
        })
        
        # 检查文件是否存在
        if not os.path.isfile(excel_filename):
            lake_info.to_csv(excel_filename, index=False, header=True, mode='w')
        else:
            # 如果文件存在，读取现有数据并追加新数据
            existing_data = pd.read_csv(excel_filename)
            combined_data = pd.concat([existing_data, lake_info], ignore_index=True)
            combined_data.to_csv(excel_filename, index=False, header=True, mode='w')
                
        pio.write_html(fig, file= f"lake_{lake_sort}_{ex_num}_plot.html", auto_open=False)
        # pio.write_image(fig, f"lake_{lake_sort}_{ex_num}_plot.png", scale=1, width=300, height=200, format='png')
    else:
        stable_centroids = find_representative_point(lake_geom)
        x_center, y_center = stable_centroids[0].x, stable_centroids[0].y
        intersections,nears = find_intersection_with_center(x_center, y_center, x_obs, y_obs, x_b, y_b)
        intersect_points = [np.array([point.x, point.y]) for intersection in intersections for point in intersection]

        # 确保intersect_points的长度是偶数
        assert len(intersect_points) % 2 == 0

        points_on_section = gpd.GeoDataFrame(geometry=[Point(xy) for xy in zip(x_obs, y_obs)], crs="EPSG:4326").to_crs(epsg=8859)
        df = pd.DataFrame({
            'x_obs': x_obs,
            'y_obs': y_obs,
            'z_obs': z_obs,
            'obs_geo': points_on_section.geometry
        })
        df['intersect_points_0'] = intersect_points[::2]
        df['intersect_points_1'] = intersect_points[1::2]
        df['intersect_points_near'] = nears
        df = df.sort_values(by=['z_obs'], ascending=[False])
        calculate_distances_to_intersections(df, x_center, y_center,  projection_crs="EPSG:8859")
        height_shrink = calculate_shrink_boundary(df,all_boundary_points)
        shrunk_points = calculate_shrunk_boundary_points(all_boundary_points, height_shrink, x_center, y_center)
        
        shrunk_points = remove_excessive_or_intersecting_boundaries(shrunk_points,lake_geom)
        if len(shrunk_points) == 0:
            print("No shrunk points found.")
            continue
        
        z_center,x_interp,y_interp,z_interp= calculate_center_depth(x_center, y_center, z_obs,
                                                                    shrunk_points, lake_geom, lake_level,plot=False)
        x_interps,y_interps,z_interps = calculate_others(x_center, y_center, z_center, z_obs, shrunk_points,lake_geom, lake_level)

        x_interp = np.concatenate((x_interp, x_interps))
        y_interp = np.concatenate((y_interp, y_interps))
        z_interp = np.concatenate((z_interp, z_interps))
        
        xi,yi,zi = RBF_interpolation(shrunk_points,x_b,y_b,z_b,x_interp,y_interp,z_interp,x_center,y_center,z_center)
        lake_volume, lake_maxdeep, lake_meandeep,lake_width,lake_length = calculate_lake_volume(xi, yi, zi,lake_geom,lake_level,
                                                                        crs="EPSG:4326",projection_crs="EPSG:8859")
        
        
        fig = plot_lake_surface(xi, yi, zi, lake_level, x_center, y_center,z_center,x_interp,y_interp,z_interp,
                    x_b=x_b, y_b=y_b, z_b=z_b, 
                    x_obs=x_obs, y_obs=y_obs, z_obs=z_obs,
                    shrunk_points=shrunk_points, df=df,
                    add_boundary_points=True, add_spline_interp = True,
                    add_center_point=True, 
                    add_center_with_depth=True,add_observation_points=True, 
                    add_boundary_lines=True, add_intersection_lines=True,show=False)

        
        excel_filename = f"lake_data.csv"
        os.chdir(r'G:\SETP_ICESat-2\lake_volumn_by_ICEsat-2\裂化')
        # 记录湖泊数据
        lake_info = pd.DataFrame({
            'lake_sort': [lake_sort],
            'lake_level': [lake_level],
            'stable_centroids': [str(stable_centroids[0].coords[0])],  # 将Point对象转换为字符串
            'lake_volume': [lake_volume],
            'lake_maxdeep': [lake_maxdeep],
            'lake_meandeep': [lake_meandeep],
            'lake_width': [lake_width],
            'lake_length': [lake_length]
        })
        
        # 检查文件是否存在
        if not os.path.isfile(excel_filename):
            lake_info.to_csv(excel_filename, index=False, header=True, mode='w')
        else:
            # 如果文件存在，读取现有数据并追加新数据
            existing_data = pd.read_csv(excel_filename)
            combined_data = pd.concat([existing_data, lake_info], ignore_index=True)
            combined_data.to_csv(excel_filename, index=False, header=True, mode='w')
                
        pio.write_html(fig, file= f"lake_{lake_sort}_{ex_num}_plot.html", auto_open=False)
        # pio.write_image(fig, f"lake_{lake_sort}_{ex_num}_plot.png", scale=1, width=300, height=200, format='png')
        
        print('存在裂化') 



