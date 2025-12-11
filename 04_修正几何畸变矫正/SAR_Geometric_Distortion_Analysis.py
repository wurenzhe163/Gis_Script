# %%
# Section 1: Environment Setup and Imports
# This section initializes the Google Earth Engine (GEE) environment and imports necessary modules
# for SAR geometric distortion analysis. It configures the PROJ library path for coordinate
# transformations and establishes GEE authentication.

import ee
import geemap
import sys,os
proj_lib_path = r"C:\ProgramData\anaconda3\envs\GEE\Lib\site-packages\pyproj\proj_dir\share\proj"
os.environ['PROJ_LIB'] = proj_lib_path
from GEE_Func.S1_distor_dedicated import load_S1collection,S1_CalDistor
from GEE_Func.GEE_DataIOTrans import BandTrans,DataTrans
from GEE_Func.GEE_CorreterAndFilters import S1Corrector
from functools import partial
Eq_pixels = DataTrans.Eq_pixels
# geemap.set_proxy(port=10809)
ee.Authenticate()
ee.Initialize()

# %%
# Section 2: Core Geometric Processing Functions
# This section contains the fundamental algorithms for SAR geometric distortion analysis.
# Key functions include:
# - Line2Points: Converts line geometries to interpolated point collections for analysis
# - get_neighborhood_info: Extracts spatial neighborhood data (4-point or 9-point configurations)
# - Surface modeling functions: Volum9_func (quadratic fitting), Flat4_func (planar fitting), 
#   Bilinear_interp_func (bilinear interpolation)
# - Main_CalNeighbor: Orchestrates neighborhood-based elevation and angle calculations

import numpy as np
from scipy.optimize import least_squares

def Line2Points(feature,region,scale=30):
    line_geometry = ee.Feature(feature).geometry().intersection(region, maxError=1)
    coordinates = line_geometry.coordinates()
    start_point = ee.List(coordinates.get(0))
    end_point = ee.List(coordinates.get(-1))
    length = line_geometry.length()
    num_points = length.divide(scale).floor()
    
    def interpolate(i):
        i = ee.Number(i)
        fraction = i.divide(num_points)
        interpolated_lon = ee.Number(start_point.get(0)).add(
            ee.Number(end_point.get(0)).subtract(ee.Number(start_point.get(0))).multiply(fraction))
        interpolated_lat = ee.Number(start_point.get(1)).add(
            ee.Number(end_point.get(1)).subtract(ee.Number(start_point.get(1))).multiply(fraction))
        return ee.Feature(ee.Geometry.Point([interpolated_lon, interpolated_lat]))

    filtered_points = ee.FeatureCollection(ee.Algorithms.If(num_points.gt(0), 
                                                            ee.FeatureCollection(ee.List.sequence(1, num_points).map(interpolate)),
                                                            ee.FeatureCollection([])))
    return filtered_points

def filter_Listlen(item,len=3):
    item_list = ee.List(item)
    return ee.Algorithms.If(item_list.size().gte(len), item_list, None)

def get_neighborhood_info(point,image,scale=15,reduceSacle=30,neighborhood_type='4',Pointfilter=False):
    point = ee.Geometry.Point(point)
    buffer_region = point.buffer(scale)
    region = buffer_region.bounds()
    coords = ee.List(region.coordinates().get(0))
    corner1 = ee.Geometry.Point(ee.List(coords.get(0)))
    corner2 = ee.Geometry.Point(ee.List(coords.get(1)))
    corner3 = ee.Geometry.Point(ee.List(coords.get(2)))
    corner4 = ee.Geometry.Point(ee.List(coords.get(3)))
    
    if neighborhood_type=='4':
        region_coord = ee.Geometry.MultiPoint([
                                            corner1.coordinates(), corner2.coordinates(), 
                                            corner3.coordinates(), corner4.coordinates()])
    
    elif neighborhood_type=='9': 
        edge_center1 = ee.Geometry.LineString([corner1.coordinates(), corner2.coordinates()]).centroid()
        edge_center2 = ee.Geometry.LineString([corner2.coordinates(), corner3.coordinates()]).centroid()
        edge_center3 = ee.Geometry.LineString([corner3.coordinates(), corner4.coordinates()]).centroid()
        edge_center4 = ee.Geometry.LineString([corner4.coordinates(), corner1.coordinates()]).centroid()
        region_coord = ee.Geometry.MultiPoint([
                        corner1.coordinates(), corner2.coordinates(), 
                        corner3.coordinates(), corner4.coordinates(),
                        edge_center1.coordinates(), edge_center2.coordinates(), 
                        edge_center3.coordinates(), edge_center4.coordinates(),
                        point.coordinates()])
    else:
        print('neighborhood_type should be "4" or "9"')

    result = image.reduceRegion(
        reducer=ee.Reducer.toList(),
        geometry=region_coord,
        scale=reduceSacle,
        maxPixels=1e9)
    result = result.set('point_coordinates', point.coordinates())
    if Pointfilter:
        result = ee.Algorithms.If(ee.List(result.get('angle')).size().gte(int(neighborhood_type)), result, None)
    return result 

def weighted_avg_func(neighbor):
    neighbors_info = ee.Dictionary(neighbor)
    lon_coords = ee.Array(neighbors_info.get('longitude'))
    lat_coords = ee.Array(neighbors_info.get('latitude'))
    elevations = ee.Array(neighbors_info.get('elevation'))
    x_ = ee.Array(neighbors_info.get('x'))
    y_ = ee.Array(neighbors_info.get('y'))
    angles = ee.Array(neighbors_info.get('angle'))
    point_coords = ee.List(neighbors_info.get('point_coordinates'))
    
    point1_x = ee.Number(point_coords.get(0))
    point1_y = ee.Number(point_coords.get(1))
    distances = lon_coords.subtract(point1_x).pow(2).add(lat_coords.subtract(point1_y).pow(2)).sqrt()
    weights = distances.pow(-1)
    sum_weights = weights.reduce(ee.Reducer.sum(), [0])

    weighted_elevations = elevations.multiply(weights)
    weighted_avg_elevation = weighted_elevations.reduce(ee.Reducer.sum(), [0]).divide(sum_weights).get([0])

    weighted_angles = angles.multiply(weights)
    weighted_avg_angles = weighted_angles.reduce(ee.Reducer.sum(), [0]).divide(sum_weights).get([0])
    
    weighted_X = x_.multiply(weights)
    weighted_avg_X = weighted_X.reduce(ee.Reducer.sum(), [0]).divide(sum_weights).get([0])
    
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
    def equation(params, x, y):
        a, b, c, d, e, f = params
        return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

    def residuals(params, x, y, z_true):
        z_pred = equation(params, x, y)
        return z_pred - z_true
    
    def calculate_z(params, x, y):
        a, b, c, d, e, f = params
        z_pred = a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
        return z_pred
    
    lon = np.array(neighbors_info['longitude'])
    lat = np.array(neighbors_info['latitude'])
    elevation = np.array(neighbors_info['elevation'])
    angles = np.array(neighbors_info['angle'])
    x_ = np.array(neighbors_info['x'])
    y_ = np.array(neighbors_info['y'])
    lon_pre, lat_pre = np.array(neighbors_info['point_coordinates'])

    initial_guess = np.zeros(6)

    result0 = least_squares(residuals, initial_guess, args=(lon, lat, elevation))
    params0 = result0.x  
    z_pred = calculate_z(params0, lon_pre, lat_pre)

    result1 = least_squares(residuals, initial_guess, args=(lon, lat, angles))
    params1 = result1.x  
    angle_pred = calculate_z(params1, lon_pre, lat_pre)
    
    result2 = least_squares(residuals, initial_guess, args=(lon, lat, x_))
    params2 = result2.x  
    x_pred = calculate_z(params2, lon_pre, lat_pre)
    
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
    
    A = np.vstack([lon, lat, np.ones(len(lon))]).T
    b = elevation
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
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

    distances = np.sqrt((lon - lon_pred)**2 + (lat - lat_pred)**2)
    idx = np.argsort(distances)[:4]
    lon = lon[idx]
    lat = lat[idx]
    elevation = elevation[idx]
    angles = angles[idx]
    x_ = x_[idx]
    y_ = y_[idx]

    def bilinear_interp(values):
        matrix = np.array([
            [1, lon[0], lat[0], lon[0] * lat[0]],
            [1, lon[1], lat[1], lon[1] * lat[1]],
            [1, lon[2], lat[2], lon[2] * lat[2]],
            [1, lon[3], lat[3], lon[3] * lat[3]]
        ])
        coeffs = np.linalg.solve(matrix, values)
        return coeffs[0] + coeffs[1] * lon_pred + coeffs[2] * lat_pred + coeffs[3] * lon_pred * lat_pred

    z_pred = bilinear_interp(elevation)
    angle_pred = bilinear_interp(angles)
    x_pred = bilinear_interp(x_)
    y_pred = bilinear_interp(y_)

    return {'elevation': z_pred, 'angle': angle_pred, 'x': x_pred, 'y': y_pred, 'point_coordinates': [lon_pred, lat_pred]}

def apply_map_to_list(input_list, func): return input_list.map(func)

def apply_map_to_list_local(input_list, func): return map(func,input_list)

def Main_CalNeighbor(Templist_A,AOI,Prj_scale,Cal_image,Neighbors='4',Elvevation_model='weighted_avg_elevation'):
    Len_Templist_A =  Templist_A.size().getInfo()
    Neighbors = Neighbors
    Elvevation_model = Elvevation_model

    All_PointLine = [] 
    pbar = range(Len_Templist_A)
    for i in pbar:
        points = Line2Points(Templist_A.get(i),region=AOI,scale=Prj_scale)  
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
        neighbors_info = cal_neighbors.getInfo()
        if Elvevation_model == 'Area_elavation':
            Points_WithH_Angle = list(apply_map_to_list_local(neighbors_info,lambda x: list(map(Flat4_func,x))))
        elif Elvevation_model == 'Volum_elavation':
            Points_WithH_Angle = list(apply_map_to_list_local(neighbors_info,lambda x: list(map(Volum9_func,x))))
        elif Elvevation_model == 'Bilinear_interp':
            Points_WithH_Angle = list(apply_map_to_list_local(neighbors_info,lambda x: list(map(Bilinear_interp_func,x))))
    return Points_WithH_Angle

# %%
# Section 3: Extreme Point Detection and Analysis
# This section implements algorithms for identifying critical terrain features that cause
# geometric distortions in SAR imagery. Key functions include:
# - compute_derivative_same_length: Calculates terrain derivatives using cubic spline interpolation
# - compute_curvature: Computes terrain curvature from first and second derivatives
# - find_local_extrema_indices: Identifies local maxima and minima in elevation profiles
# - filter_local_maxima_by_elevation_difference: Filters peaks based on elevation thresholds
# - calculate_extreme_points_for_segment: Combines multiple criteria to identify distortion-causing features
# - Main_CalMaxPoints: Handles orbit direction (ascending/descending) for proper peak indexing

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline

def compute_derivative_same_length(elev_list,nu=1):
    spline = CubicSpline(range(len(elev_list)), elev_list)
    derivative = spline.derivative(nu=nu)(range(len(elev_list)))
    return derivative

def compute_curvature(first_deriv, second_deriv):
    curvature = np.abs(second_deriv) / (1 + first_deriv**2)**1.5
    return curvature

def find_local_extrema_indices(array):
    maxima_indices, _ = find_peaks(array)
    minima_indices, _ = find_peaks(-array)
    return maxima_indices, minima_indices

def filter_local_maxima_by_elevation_difference(local_maxima_indices, local_minima_indices, elevation_list, elevation_difference_threshold=15):
    filtered_maxima_indices = []
    for maxima_index in local_maxima_indices:
        maxima_elevation = elevation_list[maxima_index]
        
        minima_before = local_minima_indices[local_minima_indices < maxima_index]
        minima_after = local_minima_indices[local_minima_indices > maxima_index]
        
        if minima_before.size > 0 and minima_after.size > 0:
            nearest_minima_before = minima_before[-1]
            nearest_minima_after = minima_after[0]
            
            if (np.abs(elevation_list[nearest_minima_before] - maxima_elevation) >= elevation_difference_threshold or
                    np.abs(elevation_list[nearest_minima_after] - maxima_elevation) >= elevation_difference_threshold):
                filtered_maxima_indices.append(maxima_index)
        else:
            filtered_maxima_indices.append(maxima_index)
    
    return np.array(filtered_maxima_indices)

def calculate_extreme_points_for_segment(elevations, elevation_difference_threshold=15):
    elevations = np.array(elevations)

    first_derivative = compute_derivative_same_length(elevations,nu=1)
    second_derivative = compute_derivative_same_length(elevations,nu=2)
    curvature = compute_curvature(first_derivative, second_derivative)
    
    elevations_local_maxima_indices, elevations_local_minima_indices = find_local_extrema_indices(elevations)
    curvature_local_maxima_indices,_ =  find_local_extrema_indices(curvature)
    
    filtered_elevations_maxima_indices = filter_local_maxima_by_elevation_difference(
        elevations_local_maxima_indices, elevations_local_minima_indices, elevations, elevation_difference_threshold)

    filtered_derivatives_indices = np.where((first_derivative >= 0) & (second_derivative <= 0) & (curvature>=0))
    intersection_indices = np.intersect1d(filtered_derivatives_indices,curvature_local_maxima_indices)

    unionFilter_indices = np.sort(np.unique(np.concatenate((intersection_indices, filtered_elevations_maxima_indices)))).astype(np.uint8)

    return unionFilter_indices,intersection_indices,elevations_local_maxima_indices,filtered_elevations_maxima_indices

def Main_CalMaxPoints(Points_WithH_Angle,Orbit):
    unionFilter_indices = [np.arange(len(each)).astype(np.uint8) for each in Points_WithH_Angle]
    if Orbit == 'DESCENDING':
        updated_data = [(points[::-1], len(points) - indices - 1) for points, indices in zip(Points_WithH_Angle, unionFilter_indices)]
        Points_WithH_Angle, unionFilter_indices = zip(*updated_data)
    return Points_WithH_Angle,unionFilter_indices

# %%
# Section 4: Geometric Distortion Feature Calculation
# This section implements the core algorithms for calculating SAR geometric distortions
# including layover, shadow, and foreshortening effects. Key functions include:
# - calculate_left_layover: Identifies left-side layover distortion features
# - calculate_right_layover: Computes right-side layover based on left layover results
# - calculate_shadow: Detects radar shadow areas behind terrain obstacles
# - calculate_forshortening: Identifies foreshortening effects in SAR imagery
# - create_feature_collection: Converts distortion features to GeoJSON format
# - Main_CalDistortion_toEE/Main_CalDistortion_local: Main distortion calculation workflows

from scipy.spatial import distance
from collections import defaultdict
import geopandas as gpd    
from shapely.geometry import Point 
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from shapely.geometry import Polygon

def calculate_distance(point1, point2, scale=10):
    return distance.euclidean(
        (point1['x'] * scale, point1['y'] * scale), 
        (point2['x'] * scale, point2['y'] * scale))

def calculate_angle(elevation_difference, distance):
    if elevation_difference > 0:
        return np.arctan2(elevation_difference,distance) * 180 / np.pi
    else: return -1

def check_distortion(arc_angle, reference_angle):
    if arc_angle >= reference_angle: return 1
    else: return 0

def calculate_left_layover(points, extreme_point, scale=10):
    features_left = []
    for point in points:
        elevation_difference = extreme_point['elevation'] - point['elevation']
        dist = calculate_distance(extreme_point, point, scale=scale)
        arc_angle = calculate_angle(elevation_difference, dist)
        distortion = check_distortion(arc_angle, extreme_point['angle'])
        if (distortion == 1) & (elevation_difference > 0):
            features_left.append({
                'elevation':point['elevation'],
                'elevation_difference': elevation_difference,
                'distance': dist,
                'arc_angle': arc_angle,
                'distortion': distortion,
                'distortion_type': 'Leftlayover',
                'first_derivative':0,
                'point_coordinates': point['point_coordinates'],
                'values': 1
            })
    return features_left

def calculate_right_layover(features_left, extreme_point, points, scale=10):
    distorted_features_left = [f for f in features_left if f['distortion'] == 1]
    if distorted_features_left:
        max_distance_feature = max(distorted_features_left, key=lambda x: x['distance'])
        max_distance_elevation = max_distance_feature['elevation_difference']

        new_extreme_point = {
            'elevation': extreme_point['elevation'] - max_distance_elevation,
            'angle': extreme_point['angle'],
            'x':extreme_point['x'],
            'y':extreme_point['y'],
            'point_coordinates': extreme_point['point_coordinates']}

        features_right = []
        for point in points:
            elevation_difference = point['elevation'] - new_extreme_point['elevation']
            elevation_difference2 = extreme_point['elevation'] - point['elevation']
            dist = calculate_distance(new_extreme_point, point, scale=scale)
            if elevation_difference > 0 and elevation_difference2 > 0:
                arc_angle = calculate_angle(elevation_difference, dist)
            else:
                arc_angle = -1
            distortion = check_distortion(arc_angle, extreme_point['angle'])
            if (distortion == 1) & (elevation_difference > 0):
                features_right.append({
                    'elevation':point['elevation'],
                    'elevation_difference': elevation_difference,
                    'distance': dist,
                    'arc_angle': arc_angle,
                    'distortion': distortion,
                    'distortion_type': 'Rightlayover',
                    'first_derivative':0,
                    'point_coordinates': point['point_coordinates'],
                    'values': 5
                })

        if any(f['distortion'] == 1 for f in features_right):
            features_right.append({
                'elevation':point['elevation'],
                'elevation_difference': 0,
                'elevation_difference2': 0,
                'distance': 0,
                'arc_angle': 0,
                'distortion': 1,
                'distortion_type': 'Rightlayover',
                'first_derivative':0,
                'point_coordinates': extreme_point['point_coordinates'],
                'values': 5
            })
        
        return features_right
    else: return []

def calculate_shadow(points, extreme_point, scale=10):
    shadow_features = []
    for point in points:
        elevation_difference = extreme_point['elevation'] - point['elevation']
        dist = calculate_distance(extreme_point, point, scale=scale)
        arc_angle = calculate_angle(elevation_difference, dist)
        distortion = check_distortion(arc_angle, 90-extreme_point['angle'])
        if (distortion == 1) & (elevation_difference > 0):
            shadow_features.append({
                'elevation':point['elevation'],
                'elevation_difference': elevation_difference,
                'distance': dist,
                'arc_angle': arc_angle,
                'distortion': distortion,
                'distortion_type': 'Shadow',
                'first_derivative':0,
                'point_coordinates': point['point_coordinates'],
                'values': 7
            })
    return shadow_features

def calculate_forshort(points):
    elevations = np.array([each['elevation'] for each in points])
    first_derivative = compute_derivative_same_length(elevations,nu=1)
    derivative_indices = np.where(first_derivative>0)
    filtered_points = [points[i] for i in derivative_indices[0]]
    filtered_first_derivative = [first_derivative[i] for i in derivative_indices[0]]

    for i in range(len(filtered_points)):
        filtered_points[i] = {
            'elevation': filtered_points[i]['elevation'],
            'elevation_difference': 999,
            'distortion': 1,  
            'distortion_type': 'Foreshortening',
            'point_coordinates': filtered_points[i]['point_coordinates'],
            'first_derivative':filtered_first_derivative[i],
            'values': 9
        }
    return filtered_points

def calculate_distortion_features(points, indices, candidate_distortion_points=10, DistanceScale=10):
    distortion_features = []
    for index in indices:
        extreme_point = points[index]
        start_index_left = max(0, index - candidate_distortion_points)
        previous_points_left = points[start_index_left:index]
        start_index_right = min(len(points) - 1, index + candidate_distortion_points)
        after_points_right = points[index + 1:start_index_right + 1][::-1]
        features_left = calculate_left_layover(previous_points_left, extreme_point, DistanceScale)
        features_right = calculate_right_layover(features_left, extreme_point, after_points_right, DistanceScale)
        shadow_features = calculate_shadow(after_points_right, extreme_point, DistanceScale)
        forshort_features = calculate_forshort(points)
        distortion_features.extend(features_left + features_right + shadow_features + forshort_features)
    return distortion_features

def create_feature_collection(distortion_features):
    features = [
        ee.Feature(
            ee.Geometry.Point(feature['point_coordinates']),
            {'values': feature['values'],
             'distortion_type':feature['distortion_type'],
             'first_derivative_max':feature['first_derivative_max']}
        )
        for feature in distortion_features
    ]
    return ee.FeatureCollection(features)

def cal_unique_features(Points_WithH_Angle, unionFilter_indices):
    processed_data = defaultdict(lambda: {
        'distortion_types': set(), 
        'total_value': 0, 
        'first_derivative_max': 0 
    })

    all_distortion_Points = []
    for points, indices in zip(Points_WithH_Angle, unionFilter_indices):
        all_distortion_Points.extend(calculate_distortion_features(points, indices, 
                                candidate_distortion_points=10, DistanceScale=10))
                                                                                                                                     
    for feature in all_distortion_Points:
        coordinates = tuple(feature['point_coordinates'])
        distortion_type = feature['distortion_type']
        value = feature['values']
        first_derivative = feature['first_derivative']
        elevation_ = feature['elevation']

        if distortion_type not in processed_data[coordinates]['distortion_types']:
            processed_data[coordinates]['distortion_types'].add(distortion_type)
            processed_data[coordinates]['total_value'] += value
            processed_data[coordinates]['elevation'] = feature['elevation']

        if first_derivative > processed_data[coordinates]['first_derivative_max']:
            processed_data[coordinates]['first_derivative_max'] = first_derivative

    unique_features = []
    for coordinates, data in processed_data.items():
        unique_features.append({
            'point_coordinates': list(coordinates),
            'distortion_type': '-'.join(sorted(data['distortion_types'])),
            'elevation':data['elevation'],
            'values': data['total_value'],
            'first_derivative_max': data['first_derivative_max']
        })
    return unique_features

def Main_CalDistortion_toEE(Points_WithH_Angle, unionFilter_indices):
    unique_features = cal_unique_features(Points_WithH_Angle, unionFilter_indices)
    distortion_features = create_feature_collection(unique_features)
    return distortion_features

def Main_CalDistortion_local(Points_WithH_Angle, unionFilter_indices,
                             AOI,
                             epsg=4326,
                             save_shape=False,
                             to_UTM=True,
                             res=30,
                             Nodata=0,
                             point_items:list=['value','first_derivative_max'],
                             dtype = rasterio.uint8,
                             save_rasters=None):
    unique_features = cal_unique_features(Points_WithH_Angle, unionFilter_indices)
    aoi = AOI.getInfo()
    aoi_polygon = Polygon(aoi['coordinates'][0])
    points = gpd.GeoDataFrame({
        'distortion_type': [feature['distortion_type'] for feature in unique_features],
        'values': [feature['values'] for feature in unique_features],
        'first_derivative_max': [feature['first_derivative_max'] for feature in unique_features],
        'geometry': [Point(feature['point_coordinates']) for feature in unique_features],
        'elevation':[feature['elevation'] for feature in unique_features]
    })
    points.set_crs(epsg=epsg, inplace=True)
    gdf_aoi = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[aoi_polygon])
    
    if save_shape:
        points.to_file("point_data.shp")

    if to_UTM:
        centroid = gdf_aoi.geometry.centroid.iloc[0]
        utm_zone = int((centroid.x + 180) / 6) + 1
        utm_crs = f'epsg:{32600 + utm_zone if centroid.y > 0 else 32700 + utm_zone}'
        gdf_aoi = gdf_aoi.to_crs(utm_crs)
        points = points.to_crs(utm_crs)

    xmin, ymin, xmax, ymax = gdf_aoi.total_bounds
    x_res = y_res = res
    width = int((xmax - xmin) / x_res)
    height = int((ymax - ymin) / y_res)
    transform = from_origin(west=xmin, north=ymax, xsize=x_res, ysize=y_res)
    out_shape = (height, width)

    rasters = [rasterize(
        ((geom, value) for geom, value in zip(points.geometry, points[point_item.replace('value', 'values')])),
        out_shape=out_shape,
        fill=Nodata,
        transform=transform,
        dtype=dtype
    ) for point_item in point_items]

    if save_rasters:
        for raster,save_raster in zip(rasters,save_rasters):
            with rasterio.open(
                save_raster,
                'w',
                driver='GTiff',
                height=raster.shape[0],
                width=raster.shape[1],
                count=1,
                dtype=raster.dtype,
                crs=points.crs,
                transform=transform,
            ) as dst:
                dst.write(raster, 1)

# %%
# Section 5: Main Execution Workflow
# This section implements the complete SAR geometric distortion analysis pipeline.
# It processes S1 SAR imagery over a fishnet grid covering the Southeast Tibet region,
# calculating geometric distortions for both ascending and descending orbits.
# Key components include:
# - Fishnet grid generation for systematic area coverage
# - S1 SAR image loading and preprocessing (ascending/descending orbits)
# - Geometric parameter calculation (azimuth angles, coordinates, heading)
# - Elevation model integration (NASADEM, ALOS, COPERNICUS DEMs)
# - Distortion processing loop with progress tracking and error handling
# - Raster output generation for distortion features and derivatives

import traceback
from tqdm import tqdm
from PackageDeepLearn.utils import DataIOTrans
from IPython.display import clear_output

Southest_doom = ee.FeatureCollection('projects/ee-mrwurenzhe/assets/ChinaShp/SouthestRegion')
Southest_doom_fishnet = geemap.fishnet(Southest_doom.first().geometry(), rows=100, cols=150, delta=1)
lenfish_net = Southest_doom_fishnet.size().getInfo()
Southest_doom_fishnet = Southest_doom_fishnet.toList(lenfish_net)

year = '2019'
START_DATE  = ee.Date(year + '-01-01')
END_DATE   = ee.Date(year + '-12-30')
TIME_LEN   = END_DATE.difference(START_DATE, 'days').abs()
MIDDLE_DATE = START_DATE.advance(TIME_LEN.divide(ee.Number(2)).int(),'days')
Origin_scale = 10
Prj_scale = 30
Savpath = DataIOTrans.make_dir(r'藏东南几何畸变数据存储')
os.chdir(Savpath)

DEMSRTM = ee.Image('USGS/SRTMGL1_003')
DEM_prj = DEMSRTM.projection()
DEMNASA = ee.Image("NASA/NASADEM_HGT/001").select('elevation')
DEMALOS = ee.ImageCollection("JAXA/ALOS/AW3D30/V4_1").mosaic().select('DSM').rename('elevation').reproject(DEM_prj)
DEMCOPERNICUS = ee.ImageCollection("COPERNICUS/DEM/GLO30").mosaic().select('DEM').rename('elevation').int16().reproject(DEM_prj)
DEM = DEMNASA
Wrong_dataIndex = []

pbar= tqdm(range(lenfish_net), ncols=80)

index_p = 0
for i in pbar:
    if i < index_p:
        continue
    if i % 100 == 0 and i != 0:
        clear_output(wait=True)
    pbar.set_description('Processing '+str(i))
    AOI = ee.Feature(Southest_doom_fishnet.get(i)).geometry()
    s1_ascending, s1_descending = load_S1collection(AOI,START_DATE,END_DATE,MIDDLE_DATE,FilterSize=30)
    for Orbit in ['ASCENDING', 'DESCENDING']:
        if Orbit == 'ASCENDING':
            S1_image = s1_ascending
        elif Orbit == 'DESCENDING':
            S1_image = s1_descending

        Projection = S1_image.select(0).projection()
        Mask = S1_image.select(0).mask()
        azimuthEdge, rotationFromNorth, startpoint, endpoint, coordinates_dict  = S1Corrector.getS1Corners(S1_image, AOI, Orbit) 
        Heading = azimuthEdge.get('azimuth')
        s1_azimuth_across = ee.Number(Heading).subtract(90.0)
        Auxiliarylines = ee.Geometry.LineString([startpoint, endpoint])

        BandTrans.delBands(S1_image, ['VH', 'VV'])
        Cal_image = (Eq_pixels(BandTrans.delBands(S1_image, 'VV','VH').resample('bicubic')).rename('angle')
                            .addBands(ee.Image.pixelCoordinates(Projection))
                            .addBands(DEM.select('elevation'))
                            .addBands(ee.Image.pixelLonLat())
                            .updateMask(Mask)
                            .reproject(crs=Projection, scale=Prj_scale)
                            .clip(AOI))
        try:
            Projection = Cal_image.select(0).projection()
            Templist = S1_CalDistor.AuxiliaryLine2Point(s1_azimuth_across, coordinates_dict, Auxiliarylines,AOI, Prj_scale)
            pbar.set_description('Completed Templist')

            Points_WithH_Angle = Main_CalNeighbor(Templist,AOI,Prj_scale,Cal_image,Neighbors='4',Elvevation_model='weighted_avg_elevation')
            pbar.set_description('Completed Points_WithH_Angle')

            Points_WithH_Angle, unionFilter_indices = Main_CalMaxPoints(Points_WithH_Angle,Orbit)

            # GEE
            # distortion_features = Main_CalDistortion_toEE(Points_WithH_Angle, unionFilter_indices)
            # pbar.set_description('Completed distortion_features')

            # Distortion = ee.Image().paint(distortion_features, 'values').reproject(crs=Projection, scale=Prj_scale).clip(AOI).toInt8()
            # DataIO.Geemap_export(f'{i:06d}'+'_' +'Distortion_'+ Orbit+ '.tif',Distortion,region=AOI,scale=Prj_scale,rename_image=False)

            # First_derivative = ee.Image().paint(distortion_features, 'first_derivative_max').reproject(crs=Projection, scale=Prj_scale).clip(AOI).toInt8()
            # DataIO.Geemap_export(f'{i:06d}'+'_'+'First_derivative_'+ Orbit+ '.tif',First_derivative,region=AOI,scale=Prj_scale,rename_image=False)

            # GDAL
            Main_CalDistortion_local(Points_WithH_Angle, unionFilter_indices,AOI,epsg=4326,save_shape=False,to_UTM=True,
                            point_items=['value','first_derivative_max'],res=30,
                            save_rasters=[f'{i:06d}'+'_' +'Distortion'+ Orbit+ '.tif',f'{i:06d}'+'_'+'First_derivative'+ Orbit+ '.tif'])

        except:
            Wrong_dataIndex.append(i)
            with open('log.txt', 'a') as f: 
                f.write('Wrong index = {}\n'.format(i))
                f.write(traceback.format_exc())
                f.write('\n')
            print('Error recorded to log.txt')