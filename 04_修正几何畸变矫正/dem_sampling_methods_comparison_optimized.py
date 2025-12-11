#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DEM Sampling Methods Comparison Tool
================================================================================

This tool implements multiple DEM sampling methods comparison analysis based on 
Google Earth Engine (GEE) platform.

Key Features:
1. Sentinel-1 SAR image data loading and preprocessing
2. DEM data sampling and reconstruction
3. Multiple sampling algorithm implementations (weighted average, volume fitting, 
   bilinear interpolation, etc.)
4. Results visualization and export

Dependencies:
- earthengine-api (GEE)
- geemap
- numpy
- scipy
- geopandas
- shapely
- tqdm

Author: wrz
Created: 2024
Version: v2.0 (Optimized)
================================================================================
"""

import ee
import os
from functools import partial
import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point
from typing import Dict, List, Optional, Union, Any
import warnings
import argparse

warnings.filterwarnings('ignore')

# ================================================================================
# Module Imports and Initialization
# ================================================================================

try:
    from GEE_Func.S1_distor_dedicated import load_S1collection, S1_CalDistor
    from GEE_Func.GEE_DataIOTrans import DataTrans, BandTrans
    from GEE_Func.GEE_CorreterAndFilters import S1Corrector
    from GEE_Func.GEEMath import get_minmax
    from GEE_Func.GEE_Tools import Select_imageNum
    print('✓ GEE_Func modules imported successfully')
    
    Eq_pixels = DataTrans.Eq_pixels
    
except ImportError as e:
    print(f"✗ Module import failed: {e}")
    raise

# Initialize Google Earth Engine
try:
    ee.Authenticate()
    ee.Initialize()
    print('✓ GEE initialized successfully')
except Exception as e:
    print(f"✗ GEE initialization failed: {e}")
    print("Please check network connection and proxy settings")
    raise

# ================================================================================
# Geometry Processing Functions
# ================================================================================

def line_to_points(feature: ee.Feature, region: ee.Geometry, scale: int = 30) -> ee.FeatureCollection:
    """
    Convert line feature to equally spaced point sequence
    
    Args:
        feature: Input line feature
        region: Study area boundary
        scale: Sampling interval (meters)
        
    Returns:
        FeatureCollection: Point feature collection
    """
    line_geometry = ee.Feature(feature).geometry().intersection(region, maxError=1)
    
    coordinates = line_geometry.coordinates()
    start_point = ee.List(coordinates.get(0))
    end_point = ee.List(coordinates.get(-1))
    
    length = ee.Number(line_geometry.length())
    num_points = length.divide(ee.Number(scale)).floor()
    
    def interpolate_point(index: ee.Number) -> ee.Feature:
        fraction = ee.Number(index).divide(ee.Number(num_points))
        
        start_lon = ee.Number(start_point.get(0))
        end_lon = ee.Number(end_point.get(0))
        interpolated_lon = start_lon.add(end_lon.subtract(start_lon).multiply(fraction))
        
        start_lat = ee.Number(start_point.get(1))
        end_lat = ee.Number(end_point.get(1))
        interpolated_lat = start_lat.add(end_lat.subtract(start_lat).multiply(fraction))
        
        return ee.Feature(ee.Geometry.Point([interpolated_lon, interpolated_lat]))
    
    return ee.FeatureCollection(ee.List.sequence(1, ee.Number(num_points).max(1)).map(interpolate_point))

def filter_list_length(item: ee.List, min_len: int = 3) -> Optional[ee.List]:
    """Filter lists shorter than specified length"""
    item_list = ee.List(item)
    return ee.Algorithms.If(item_list.size().gte(min_len), item_list, None)

# ================================================================================
# Neighborhood Information Extraction Functions
# ================================================================================

def get_neighborhood_info(point: List[float], 
                         image: ee.Image,
                         scale: int = 15,
                         reduce_scale: int = 30,
                         neighborhood_type: str = '4',
                         point_filter: bool = False) -> Optional[ee.Dictionary]:
    """
    Extract remote sensing image information within point neighborhood
    
    Args:
        point: Target point coordinates [lon, lat]
        image: Input remote sensing image
        scale: Neighborhood range size (meters)
        reduce_scale: Resampling scale (meters)
        neighborhood_type: Neighborhood type ('4' or '9')
        point_filter: Whether to filter neighborhoods with insufficient points
        
    Returns:
        Dictionary containing neighborhood information or None
    """
    point_geom = ee.Geometry.Point(point)
    buffer_region = point_geom.buffer(scale)
    
    bounds = buffer_region.bounds()
    coords = ee.List(bounds.coordinates().get(0))
    
    corners = [
        ee.Geometry.Point(ee.List(coords.get(i)))
        for i in range(4)
    ]
    
    if neighborhood_type == '4':
        sample_points = ee.Geometry.MultiPoint([
            corners[i].coordinates() for i in range(4)
        ])
        
    elif neighborhood_type == '9':
        edge_centers = [
            ee.Geometry.LineString([corners[i].coordinates(), corners[(i+1)%4].coordinates()]).centroid()
            for i in range(4)
        ]
        
        sample_points = ee.Geometry.MultiPoint([
            corners[0].coordinates(), corners[1].coordinates(),
            corners[2].coordinates(), corners[3].coordinates(),
            edge_centers[0].coordinates(), edge_centers[1].coordinates(),
            edge_centers[2].coordinates(), edge_centers[3].coordinates(),
            point_geom.coordinates()
        ])
    else:
        raise ValueError(f"Unsupported neighborhood type: {neighborhood_type}, choose '4' or '9'")
    
    result = image.reduceRegion(
        reducer=ee.Reducer.toList(),
        geometry=sample_points,
        scale=reduce_scale,
        maxPixels=1e9
    )
    
    result = result.set('point_coordinates', point_geom.coordinates())
    
    if point_filter:
        result = ee.Algorithms.If(
            ee.List(result.get('angle')).size().gte(int(neighborhood_type)),
            result,
            None
        )
    
    return result

# ================================================================================
# Elevation Reconstruction Algorithms
# ================================================================================

def weighted_avg_func(neighbor: ee.Dictionary) -> ee.Dictionary:
    """
    Weighted average elevation reconstruction using inverse distance weighting
    Weights are distance inverse: closer distances have larger weights
    """
    neighbors_info = ee.Dictionary(neighbor)
    
    # Extract neighborhood data
    lon_coords = ee.Array(neighbors_info.get('longitude'))
    lat_coords = ee.Array(neighbors_info.get('latitude'))
    elevations = ee.Array(neighbors_info.get('elevation'))
    angles = ee.Array(neighbors_info.get('angle'))
    x_coords = ee.Array(neighbors_info.get('x'))
    y_coords = ee.Array(neighbors_info.get('y'))
    point_coords = ee.List(neighbors_info.get('point_coordinates'))
    
    # Calculate distance weights
    point_lon = ee.Number(point_coords.get(0))
    point_lat = ee.Number(point_coords.get(1))
    
    distances = lon_coords.subtract(point_lon).pow(2).add(
        lat_coords.subtract(point_lat).pow(2)
    ).sqrt()
    
    weights = distances.pow(-1)  # Distance inverse as weights
    sum_weights = weights.reduce(ee.Reducer.sum(), [0])
    
    # Calculate weighted average
    def calc_weighted_avg(values: ee.Array) -> ee.Number:
        weighted_values = values.multiply(weights)
        return weighted_values.reduce(ee.Reducer.sum(), [0]).divide(sum_weights).get([0])
    
    return ee.Dictionary({
        'elevation': calc_weighted_avg(elevations),
        'angle': calc_weighted_avg(angles),
        'x': calc_weighted_avg(x_coords),
        'y': calc_weighted_avg(y_coords),
        'point_coordinates': neighbors_info.get('point_coordinates')
    })

def avg_func(neighbor: ee.Dictionary) -> ee.Dictionary:
    """Simple average elevation reconstruction"""
    neighbors_info = ee.Dictionary(neighbor)
    
    def calc_avg(values: ee.Array) -> ee.Number:
        return values.reduce(ee.Reducer.mean(), [0]).get([0])
    
    return ee.Dictionary({
        'elevation': calc_avg(ee.Array(neighbors_info.get('elevation'))),
        'angle': calc_avg(ee.Array(neighbors_info.get('angle'))),
        'x': calc_avg(ee.Array(neighbors_info.get('x'))),
        'y': calc_avg(ee.Array(neighbors_info.get('y'))),
        'point_coordinates': neighbors_info.get('point_coordinates')
    })

def Volum9_func(neighbors_info: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Volume fitting elevation reconstruction (9-parameter model)
    Using quadratic polynomial fitting: z = ax² + by² + cxy + dx + ey + f
    """
    def equation(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        a, b, c, d, e, f = params
        return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
    
    def residuals(params: np.ndarray, x: np.ndarray, y: np.ndarray, z_true: np.ndarray) -> np.ndarray:
        return equation(params, x, y) - z_true
    
    # Extract data
    lon = np.array(neighbors_info['longitude'])
    lat = np.array(neighbors_info['latitude'])
    lon_pred, lat_pred = np.array(neighbors_info['point_coordinates'])
    
    # Fit each variable
    def fit_variable(values: np.ndarray) -> float:
        initial_guess = np.zeros(6)
        result = least_squares(residuals, initial_guess, args=(lon, lat, values))
        return equation(result.x, lon_pred, lat_pred)
    
    return {
        'elevation': fit_variable(np.array(neighbors_info['elevation'])),
        'angle': fit_variable(np.array(neighbors_info['angle'])),
        'x': fit_variable(np.array(neighbors_info['x'])),
        'y': fit_variable(np.array(neighbors_info['y'])),
        'point_coordinates': [lon_pred, lat_pred]
    }

def Flat4_func(neighbors_info: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Plane fitting elevation reconstruction (4-parameter model)
    Using linear plane fitting: z = ax + by + c
    """
    lon = np.array(neighbors_info['longitude'])
    lat = np.array(neighbors_info['latitude'])
    lon_pred, lat_pred = np.array(neighbors_info['point_coordinates'])
    
    # Build design matrix
    A = np.vstack([lon, lat, np.ones(len(lon))]).T
    
    def fit_variable(values: Union[List[float], np.ndarray]) -> float:
        b = np.array(values)
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        a, b_coeff, c = coeffs
        return a * lon_pred + b_coeff * lat_pred + c
    
    return {
        'elevation': fit_variable(neighbors_info['elevation']),
        'angle': fit_variable(neighbors_info['angle']),
        'x': fit_variable(neighbors_info['x']),
        'y': fit_variable(neighbors_info['y']),
        'point_coordinates': [lon_pred, lat_pred]
    }

def Bilinear_interp_func(neighbors_info: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Bilinear interpolation elevation reconstruction
    Select nearest 4 points for bilinear interpolation
    """
    lon = np.array(neighbors_info['longitude'])
    lat = np.array(neighbors_info['latitude'])
    lon_pred, lat_pred = np.array(neighbors_info['point_coordinates'])
    
    # Select nearest 4 points
    distances = np.sqrt((lon - lon_pred)**2 + (lat - lat_pred)**2)
    idx = np.argsort(distances)[:4]
    
    lon_near = lon[idx]
    lat_near = lat[idx]
    
    # Build interpolation matrix
    matrix = np.array([
        [1, lon_near[0], lat_near[0], lon_near[0] * lat_near[0]],
        [1, lon_near[1], lat_near[1], lon_near[1] * lat_near[1]],
        [1, lon_near[2], lat_near[2], lon_near[2] * lat_near[2]],
        [1, lon_near[3], lat_near[3], lon_near[3] * lat_near[3]]
    ])
    
    def bilinear_interp(values: np.ndarray) -> float:
        b = values[idx]
        coeffs = np.linalg.solve(matrix, b)
        return coeffs[0] + coeffs[1]*lon_pred + coeffs[2]*lat_pred + coeffs[3]*lon_pred*lat_pred
    
    return {
        'elevation': bilinear_interp(np.array(neighbors_info['elevation'])),
        'angle': bilinear_interp(np.array(neighbors_info['angle'])),
        'x': bilinear_interp(np.array(neighbors_info['x'])),
        'y': bilinear_interp(np.array(neighbors_info['y'])),
        'point_coordinates': [lon_pred, lat_pred]
    }

# ================================================================================
# Batch Processing Functions
# ================================================================================

def apply_map_to_list(input_list: ee.List, func) -> ee.List:
    """GEE list mapping function"""
    return input_list.map(func)

def apply_map_to_list_local(input_list: List, func) -> map:
    """Local list mapping function"""
    return map(func, input_list)

def main_calculate_neighbor(Templist: ee.List,
                           AOI: ee.Geometry,
                           Prj_scale: int,
                           Cal_image: ee.Image,
                           Neighbors: str = '4',
                           Elevation_model: str = 'weighted_avg_elevation') -> List[List[Dict]]:
    """
    Main calculation function: perform neighborhood sampling and elevation reconstruction for auxiliary line points
    
    Args:
        Templist: Auxiliary line list
        AOI: Study area
        Prj_scale: Projection scale
        Cal_image: Calculation image
        Neighbors: Neighborhood type ('4' or '9')
        Elevation_model: Elevation reconstruction algorithm
        
    Returns:
        Reconstruction results list
    """
    # Get list length
    list_length = Templist.size().getInfo()
    
    # Progress bar display
    pbar = tqdm(range(list_length), desc="Processing auxiliary lines")
    
    # Convert all auxiliary lines to point sequences
    all_point_lines = []
    for i in pbar:
        points = line_to_points(Templist.get(i), region=AOI, scale=Prj_scale)
        # Extract geometry coordinates from FeatureCollection
        coords = points.toList(points.size()).map(lambda f: ee.Feature(f).geometry().coordinates())
        all_point_lines.append(coords)
    
    # Convert to GEE list and filter
    ee_point_lines = ee.List(all_point_lines)
    ee_point_lines = ee_point_lines.map(partial(filter_list_length, min_len=3)).removeAll([None])
    
    # Get neighborhood information
    cal_neighbors = apply_map_to_list(
        ee_point_lines,
        lambda x: ee.List(x).map(
            partial(get_neighborhood_info,
                   image=Cal_image,
                   scale=Prj_scale // 2 if Neighbors == '4' else Prj_scale,
                   reduce_scale=Prj_scale,
                   neighborhood_type=Neighbors,
                   point_filter=False)
        ).removeAll([None])
    )
    cal_neighbors = cal_neighbors.map(partial(filter_list_length, min_len=3)).removeAll([None])
    
    # Perform elevation reconstruction using selected algorithm
    print(f"Performing elevation reconstruction using {Elevation_model} algorithm...")
    
    if Elevation_model == 'weighted_avg_elevation':
        points_with_h_angle = apply_map_to_list(
            cal_neighbors,
            lambda x: ee.List(x).map(weighted_avg_func)
        ).getInfo()
        
    elif Elevation_model == 'avg_elevation':
        points_with_h_angle = apply_map_to_list(
            cal_neighbors,
            lambda x: ee.List(x).map(avg_func)
        ).getInfo()
        
    elif Elevation_model in ['Area_elavation', 'Volum_elavation', 'Bilinear_interp']:
        neighbors_info = cal_neighbors.getInfo()
        
        if Elevation_model == 'Area_elavation':
            points_with_h_angle = [
                list(map(Flat4_func, line_points))
                for line_points in neighbors_info
            ]
        elif Elevation_model == 'Volum_elavation':
            points_with_h_angle = [
                list(map(Volum9_func, line_points))
                for line_points in neighbors_info
            ]
        elif Elevation_model == 'Bilinear_interp':
            points_with_h_angle = [
                list(map(Bilinear_interp_func, line_points))
                for line_points in neighbors_info
            ]
    else:
        raise ValueError(f"Unsupported algorithm: {Elevation_model}")
    
    return points_with_h_angle

# ================================================================================
# Main Functions
# ================================================================================

def create_sample_data() -> tuple:
    """Create sample data"""
    print("Creating sample data...")
    
    # Load test data
    domain_distor_test = ee.FeatureCollection('projects/ee-mrwurenzhe/assets/Test/DEM_ReConstruct')
    mountain = Select_imageNum(domain_distor_test, 0)
    dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation')
    
    # Set time parameters
    year = '2019'
    start_date = ee.Date(f'{year}-01-01')
    end_date = ee.Date(f'{year}-12-30')
    time_len = end_date.difference(start_date, 'days').abs()
    middle_date = start_date.advance(time_len.divide(ee.Number(2)).int(), 'days')
    
    return mountain, dem, start_date, end_date, middle_date

def setup_working_directory() -> str:
    """Setup working directory - create output folder under current script path"""
    # Get current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output folder name (based on current time and algorithm type)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder_name = f"DEM_Sampling_Results_{timestamp}"
    
    # Build complete output path
    save_path = os.path.join(current_dir, output_folder_name)
    
    # Create directory (if not exists)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"✓ Created output directory: {save_path}")
    else:
        print(f"✓ Output directory already exists: {save_path}")
    
    # Change to output directory
    os.chdir(save_path)
    print(f"✓ Working directory set to: {save_path}")
    
    return save_path

def process_s1_data(mountain: ee.Feature,
                    start_date: ee.Date,
                    end_date: ee.Date,
                    middle_date: ee.Date) -> tuple:
    """Process Sentinel-1 data"""
    print("Processing Sentinel-1 data...")
    
    aoi = ee.Feature(mountain).geometry()
    s1_ascending, s1_descending = load_S1collection(aoi, start_date, end_date, middle_date, FilterSize=30)
    
    orbit = 'ASCENDING'
    s1_image = s1_ascending
    projection = s1_image.select(0).projection()
    mask = s1_image.select(0).mask()
    
    # Get auxiliary line information
    azimuth_edge, rotation_from_north, startpoint, endpoint, coordinates_dict = S1Corrector.getS1Corners(
        s1_image, aoi, orbit
    )
    
    heading = azimuth_edge.get('azimuth')
    s1_azimuth_across = ee.Number(heading).subtract(90.0)
    auxiliary_lines = ee.Geometry.LineString([startpoint, endpoint])
    
    return (s1_image, aoi, projection, mask, s1_azimuth_across,
            coordinates_dict, auxiliary_lines)

def create_calculation_image(s1_image: ee.Image,
                           dem: ee.Image,
                           projection: ee.Projection,
                           mask: ee.Image,
                           aoi: ee.Geometry,
                           prj_scale: int = 30) -> ee.Image:
    """Create calculation image"""
    print("Creating calculation image...")
    
    # Process S1 image
    s1_image = BandTrans.delBands(s1_image, ['VH', 'VV'])
    
    # Build calculation image
    calc_image = (Eq_pixels(BandTrans.delBands(s1_image, 'VV', 'VH').resample('bicubic')).rename('angle')
                 .addBands(ee.Image.pixelCoordinates(projection))
                 .addBands(dem.select('elevation'))
                 .addBands(ee.Image.pixelLonLat())
                 .updateMask(mask)
                 .reproject(crs=projection, scale=prj_scale)
                 .clip(aoi))
    
    return calc_image

def results_to_geodataframe(points_with_h_angle: List[List[Dict]],
                           elevation_model: str) -> gpd.GeoDataFrame:
    """Convert results to GeoDataFrame"""
    print("Converting results to GeoDataFrame...")
    
    # Extract all points
    all_points = []
    all_angles = []
    all_elevations = []
    all_x = []
    all_y = []
    
    for line_points in points_with_h_angle:
        for point_data in line_points:
            all_points.append(Point(point_data['point_coordinates']))
            all_angles.append(point_data['angle'])
            all_elevations.append(point_data['elevation'])
            all_x.append(point_data['x'])
            all_y.append(point_data['y'])
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'angle': all_angles,
        'elevation': all_elevations,
        'x': all_x,
        'y': all_y,
        'geometry': all_points
    })
    
    gdf.set_crs(epsg=4326, inplace=True)
    
    return gdf

def save_results(gdf: gpd.GeoDataFrame, elevation_model: str) -> str:
    """Save results"""
    output_filename = f'{elevation_model}.shp'
    gdf.to_file(output_filename)
    print(f"✓ Results saved to: {output_filename}")
    return output_filename

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='DEM Sampling Analysis Tool - Different sampling methods comparison analysis based on Google Earth Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python different_sampling_methods_dem_comparison_optimized.py --scale 30 --neighbors 4 --algorithm weighted_avg_elevation
  python different_sampling_methods_dem_comparison_optimized.py --scale 50 --neighbors 9 --algorithm bilinear_interpolation
  python different_sampling_methods_dem_comparison_optimized.py --help

Algorithm Description:
  weighted_avg_elevation - Weighted average method (distance inverse weighting)
  avg_elevation         - Simple average method
  planar_4_elevation    - Planar fitting method (4 parameters)
  volumetric_9_elevation - Volume fitting method (9 parameters)
  bilinear_interpolation - Bilinear interpolation method

Neighborhood Types:
  4 - 4-neighborhood (corner points)
  9 - 9-neighborhood (corner points + edge midpoints + center point)
        """
    )
    
    # Projection scale parameter
    parser.add_argument(
        '--scale', '-s',
        type=int,
        default=30,
        help='Projection scale (meters), default 30m, affects sampling accuracy and computational efficiency'
    )
    
    # Neighborhood type parameter
    parser.add_argument(
        '--neighbors', '-n',
        type=str,
        choices=['4', '9'],
        default='4',
        help='Neighborhood type: 4 for 4-neighborhood (corners), 9 for 9-neighborhood (corners + edge midpoints + center), default 4'
    )
    
    # Elevation reconstruction algorithm parameter
    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        choices=['weighted_avg_elevation', 'avg_elevation', 'planar_4_elevation', 'volumetric_9_elevation', 'bilinear_interpolation'],
        default='weighted_avg_elevation',
        help='Elevation reconstruction algorithm selection, default weighted_avg_elevation (weighted average method)'
    )
    return parser

def main(prj_scale: int = 30, neighbors: str = '4', elevation_model: str = 'weighted_avg_elevation'):
    """Main function: Execute DEM sampling analysis
    
    Args:
        prj_scale: Projection scale (meters)
        neighbors: Neighborhood type ('4' or '9')
        elevation_model: Elevation reconstruction algorithm
    """
    print("="*60)
    print("Starting DEM sampling analysis...")
    print(f"Parameter configuration: Projection scale={prj_scale}m, Neighborhood type={neighbors}, Algorithm={elevation_model}")
    print("="*60)
    
    try:
        # Setup working directory
        setup_working_directory()
        
        # Create sample data
        mountain, dem, start_date, end_date, middle_date = create_sample_data()
        
        # Process S1 data
        (s1_image, aoi, projection, mask, s1_azimuth_across,
         coordinates_dict, auxiliary_lines) = process_s1_data(
            mountain, start_date, end_date, middle_date
        )
        
        # Create calculation image
        calc_image = create_calculation_image(
            s1_image, dem, projection, mask, aoi, prj_scale
        )
        
        # Generate auxiliary line points
        print("Generating auxiliary line points...")
        template_list = S1_CalDistor.AuxiliaryLine2Point(
            s1_azimuth_across, coordinates_dict, auxiliary_lines, aoi, prj_scale
        )
        print('✓ Template list generation completed')
        
        # Execute main calculation
        print(f"Using algorithm: {elevation_model}")
        points_with_h_angle = main_calculate_neighbor(
            template_list, aoi, prj_scale, calc_image,
            Neighbors=neighbors, Elevation_model=elevation_model
        )
        
        # Convert and save results
        gdf = results_to_geodataframe(points_with_h_angle, elevation_model)
        save_results(gdf, elevation_model)
        
        print("="*60)
        print("✓ DEM sampling analysis completed!")
        print("="*60)
        
        return gdf
        
    except Exception as e:
        print(f"✗ Error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
        return None

# ================================================================================
# Program Entry Point
# ================================================================================

if __name__ == '__main__':
    # Parse command line arguments
    parser = parse_arguments()
    args = parser.parse_args()
    
    # Execute main function with parameters
    result = main(
        prj_scale=args.scale,
        neighbors=args.neighbors,
        elevation_model=args.algorithm
    )
    
    if result is not None:
        print("Program executed successfully!")
    else:
        print("Program execution failed!")