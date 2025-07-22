%matplotlib qt
import geopandas as gpd
import pandas as pd
import os
from tqdm import tqdm
from shapely.geometry import Point, LineString, box,MultiPolygon, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from pyproj import Transformer
import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
import traceback
from matplotlib.patches import Polygon as MplPolygon

# 读取数据
os.chdir(r'E:\SETP_ICESat-2')
SETP_SHP = r"E:\SETP_Boundary.geojson"

ATL03_Water = pd.read_hdf(r"ATL_03_GlobalGeolocatedPhoton\ATL03_ALL\ATL03_Water.h5", key='df')
ATL03_ALL = pd.read_hdf(r"ATL_03_GlobalGeolocatedPhoton\ATL03_ALL\ATL03_ALL.h5", key='df')

ATL06_ALL = pd.read_hdf(r"ATL_06_Landice\ATL06_ALL\ATL06_ALL.h5", key='df')
ATL08_ALL = pd.read_hdf(r"ATL_08_LandVegetation\ATL08_ALL\ATL08_ALL.h5", key='df')
ATL13_ALL = pd.read_hdf(r"ATL_13_InlandSurfaceWaterData\ATL13_ALL\ATL13_ALL.h5", key='df')
ATL03_Noise = pd.read_hdf(r"ATL_03_GlobalGeolocatedPhoton\ATL03_Noise\ATL03_Noise_ALL.h5", key='df')

print('ATL03冰湖数量{}'.format(len(ATL03_ALL['Sort'].unique())))
print('ATL03冰湖噪声数量{}'.format(len(ATL03_Noise['Sort'].unique())))
print('ATL06冰湖数量{}'.format(len(ATL06_ALL['Sort'].unique())))
print('ATL08冰湖数量{}'.format(len(ATL08_ALL['Sort'].unique())))
print('ATL13冰湖数量{}'.format(len(ATL13_ALL['Sort'].unique())))

print("噪声值存在水体信号>=3的冰湖数量{}".format(len(ATL03_ALL[ATL03_ALL['signal_conf_combined'].apply(lambda x: x[4] >=3)]['Sort'].unique())))
print("面积大于0.1的ATL03冰湖数量{}".format(len(ATL03_ALL[ATL03_ALL.Area_pre>0.1]['Sort'].unique())))

# 加载其他数据
Sorted_ID = gpd.read_file(r'D:\BaiduSyncdisk\02_论文相关\在写\SAM冰湖\数据\2023_05_31_to_2023_09_15_样本修正_SpatialJoin.shp')
SAR_imageDir = r'D:\Dataset_and_Demo\SETP_GL\2023-05-31_to_2023-09-15'
tab10_colors = sns.color_palette("tab10")
custom_palette = {0:tab10_colors[0],1:tab10_colors[1],2:tab10_colors[2],3:tab10_colors[3],4:tab10_colors[4],5:tab10_colors[5]}

# 数据收集类
class dataCollector:
    def __init__(self, dataDF_list=[], datatype=[]):
        try:
            if 'ATL_03' in datatype:
                self.atl03 = dataDF_list[datatype.index('ATL_03')]
            if 'ATL_03Noise' in datatype:
                self.atl03noise = dataDF_list[datatype.index('ATL_03Noise')]
            if 'ATL_06' in datatype:
                self.atl06 = dataDF_list[datatype.index('ATL_06')]
            if 'ATL_08' in datatype:
                self.atl08 = dataDF_list[datatype.index('ATL_08')]
            if 'ATL_13' in datatype:
                self.atl13 = dataDF_list[datatype.index('ATL_13')]
        except IndexError as e:
            print("Error initializing dataCollector:", e)

    def plotData(self, ax=None, title='ICESat-2 Data'):
        try:
            if not hasattr(self, 'atl03') or self.atl03 is None:
                print("ATL03 data is missing.")
                return

            if ax is None:
                ax = plt.subplots()

            if hasattr(self, 'atl03') and not self.atl03.empty:
                self.atl03['signal_conf_hue'] = self.atl03['signal_conf_combined'].apply(lambda x: max(x) if isinstance(x, list) else None)
                sns.scatterplot(data=self.atl03, x='lat', y='h', hue='signal_conf_hue', palette=custom_palette, alpha=0.5, legend=True, s=10, ax=ax)
                
            if hasattr(self, 'atl03noise') and not self.atl03noise.empty:
                ax.scatter(self.atl03noise.lat, self.atl03noise.h, s=8, color='black', alpha=0.2, label='atl03noise')
                
            for attr, color, style, label in [
                ('atl06', 'C0', '-', 'ATL06'),
                ('atl08', 'C1', ':', 'ATL08'),
                ('atl13', 'C5', ':', 'ATL13')
            ]:
                if hasattr(self, attr) and not getattr(self, attr).empty:
                    ax.plot(getattr(self, attr).lat, getattr(self, attr).h, c=color, linestyle=style, label=label)

            ax.set_xlim(self.atl03.lat.min(), self.atl03.lat.max())
            ax.set_ylim(self.atl03.h.min(), self.atl03.h.max())
            ax.ticklabel_format(useOffset=False, style='plain')
            ax.set_title(title)
            ax.set_xlabel('')
            ax.set_ylabel('Elevation (m)')
            ax.legend(loc='upper right')

            return ax

        except Exception as e:
            print("Plotting ICESat-2 data was unsuccessful.")
            traceback.print_exc()

def merge_polygons(geom, expand_distance=0.0001, erode_distance=0.0001):
    def dilate(geom, distance):
        return geom.buffer(distance)

    def erode(geom, distance):
        return geom.buffer(-distance)
    dilated = dilate(geom, expand_distance)
    eroded = erode(dilated, erode_distance)
    return unary_union(eroded)

from matplotlib.gridspec import GridSpec

def plot_and_get_line(SAR_image_transformed, transform, selected_geometries, selected_area,
                      Selected_ATL03, Selected_ATL03_Noise, Selected_ATL06, Selected_ATL08, Selected_ATL13, 
                      Sort_num, date, subgroup, n, save_svg=False):
    # Set the font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    # Use GridSpec to define the layout
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    # 获取右侧子图的当前位置和大小
    pos = ax2.get_position()
    new_height = pos.height * 0.85  # 缩小到原来的80%
    ax2.set_position([pos.x0, pos.y0*1.7, pos.width, new_height])
    if SAR_image_transformed is not None:
        # Display SAR image
        show(SAR_image_transformed, ax=ax1, transform=transform, cmap='gray')
        ax1.set_title('SAR Image with Coordinates', fontsize=12)
        ax1.set_xlabel('Longitude', fontsize=12)
        ax1.set_ylabel('Latitude', fontsize=12)

        # Plot ATL03 line
        y_min, y_max = Selected_ATL03['lat'].min(), Selected_ATL03['lat'].max()
        x_min, x_max = (
            Selected_ATL03.loc[Selected_ATL03['lat'].idxmin(), 'lon'],
            Selected_ATL03.loc[Selected_ATL03['lat'].idxmax(), 'lon'],
        )
        ax1.plot([x_min, x_max], [y_min, y_max], color='red', linestyle='--', linewidth=2, label='ATL03 Line')
        
        # Handling MultiPolygon if necessary
        if isinstance(selected_geometries, MultiPolygon):
            merged_poly = merge_polygons(selected_geometries, expand_distance=0.0003, erode_distance=0.0003)
            if isinstance(merged_poly, MultiPolygon):
                merged_poly = max(merged_poly.geoms, key=lambda x: x.area)
            selected_geometries = merged_poly

        if selected_geometries.is_valid:
            mpl_polygon = MplPolygon(
                list(selected_geometries.exterior.coords),
                closed=True,
                edgecolor='blue',
                facecolor='none'
            )
            ax1.add_patch(mpl_polygon)

        # Annotate area value
        ax1.text(
            0.02, 0.98,
            f"Selected Area: {selected_area:.5f} sq. km",
            transform=ax1.transAxes,
            fontsize=10,
            color='black',
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )
        ax1.legend(loc='upper right', prop={'size': 10})

    # Prepare data for plotting in the second subplot
    dataDF_list = [Selected_ATL03, Selected_ATL03_Noise, Selected_ATL06, Selected_ATL08, Selected_ATL13]
    datatype = ['ATL_03', 'ATL_03Noise', 'ATL_06', 'ATL_08', 'ATL_13']
    s = dataCollector(dataDF_list=dataDF_list, datatype=datatype)
    s.plotData(ax=ax2, title='ICESat-2 Data')

    # Local variables for latitude and along-track distance
    lat = [y_min, y_max]
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
    # Transform coordinates
    x_min_3857, y_min_3857 = transformer.transform(y_min, x_min)
    x_max_3857, y_max_3857 = transformer.transform(y_max, x_max)

    # Calculate the distance in meters and convert to km
    xatc = [0, np.sqrt((x_max_3857 - x_min_3857)**2 + (y_max_3857 - y_min_3857)**2) / 1000]

    def lat2xatc(l):
        return xatc[0] + (l - lat[0]) * (xatc[1] - xatc[0]) / (lat[1] - lat[0])

    def xatc2lat(x):
        return lat[0] + (x - xatc[0]) * (lat[1] - lat[0]) / (xatc[1] - xatc[0])

    secax = ax2.secondary_xaxis(-0.075, functions=(lat2xatc, xatc2lat))
    secax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    secax.set_xlabel('Latitude / Along-Track Distance (km)', fontsize=12, labelpad=0)
    secax.tick_params(axis='both', which='major', labelsize=10)
    secax.ticklabel_format(useOffset=False, style='plain')
    
    # Plotting lake level line
    print("Please click on the plot to draw the lake level line (press Enter to finish)")
    lake_points = plt.ginput(n=-1, timeout=0)

    if lake_points:
        lake_points = sorted(lake_points, key=lambda x: x[0])
        lake_points_with_lon = []
        for lat_, height in lake_points:
            closest_lat_index = np.abs(s.atl03['lat'] - lat_).idxmin()
            closest_lon = s.atl03.at[closest_lat_index, 'lon']
            lake_points_with_lon.append((closest_lon, lat_, height))

        lake_geometry = LineString(lake_points_with_lon)
        
    # Plotting other feature lines
    print("Please click on the plot to draw other feature lines (press Enter to finish)")
    other_points = plt.ginput(n=-1, timeout=0)

    if other_points:
        other_points = sorted(other_points, key=lambda x: x[0])
        other_points_with_lon = []
        for lat_, height in other_points:
            closest_lat_index = np.abs(s.atl03['lat'] - lat_).idxmin()
            closest_lon = s.atl03.at[closest_lat_index, 'lon']
            other_points_with_lon.append((closest_lon, lat_, height))

        other_geometry = LineString(other_points_with_lon)

    # Create MultiLineString
    if lake_points or other_points:
        geometries = []
        properties = []
        if lake_points:
            geometries.append(lake_geometry)
            properties.append({'type': 'lake level'})
        if other_points:
            geometries.append(other_geometry)
            properties.append({'type': 'lake deep'})
        
        combined_gdf = gpd.GeoDataFrame(geometry=geometries, data=properties, crs="EPSG:4326")  
        combined_output_path = f'lake_volumn_by_ICEsat-2/{Sort_num}_{date}_{subgroup[0:4]}_{n}_lines.geojson'
        combined_gdf.to_file(combined_output_path, driver='GeoJSON')
        print(f"Lake level line and other feature lines saved as {combined_output_path}")
    else:
        print("No lines drawn, GeoJSON file not saved")

    if save_svg:
        svg_output_path = f'lake_volumn_by_ICEsat-2/{Sort_num}_{date}_{subgroup[0:4]}_{n}_plot.svg'
        plt.savefig(svg_output_path, format='svg', dpi=1200, bbox_inches='tight')
        print(f"Figure saved as SVG: {svg_output_path}")

    plt.close(fig)

from PackageDeepLearn.utils import file_search_wash as fsw

Images_cal = fsw.search_files(r'G:\SETP_ICESat-2\lake_volumn_by_ICEsat-2\image_for_cal_deep',endwith='.jpg')
Images_baseName = [os.path.basename(i) for i in Images_cal]

# 主循环
for each in Images_cal:
    Images_baseName = os.path.basename(each).split('_')
    Sort_num = float(Images_baseName[0])
    date     = Images_baseName[1]
    subgroup = Images_baseName[2] + '/'
    n = Images_baseName[3].split('.')[0]
    if os.path.exists(f'G:\SETP_ICESat-2\lake_volumn_by_ICEsat-2\{Sort_num}_{date}_{subgroup[0:4]}_{n}_lines.geojson'):
        continue

    
    # Sort_num = 162.0; date='2023-03-09'; subgroup = 'gt1l/',n=0
    # Sort_num = 24.0; date='2023-05-29'; subgroup = 'gt3r/',n=0
    # Sort_num = 6593.0; date='2023-10-13'; subgroup = 'gt1l/'; n=815 
    # Sort_num = 7412; date = '2023-09-02'; subgroup = 'gt3r/'; n=558
    # Sort_num = 318; date = '2023-02-03'; subgroup = 'gt2r/'; n=1347
    ID = int(Sorted_ID[Sorted_ID.Sort == Sort_num].ID)
    SAR_image_path = os.path.join(SAR_imageDir, f'{ID:05d}_Trans.tif')
    selected_geometries = Sorted_ID[Sorted_ID['Sort'] == Sort_num].geometry.iloc[0]
    selected_area = Sorted_ID[Sorted_ID['Sort'] == Sort_num].Area_pre.iloc[0]           
    with rasterio.open(SAR_image_path) as src:
        # 原始影像信息
        SAR_image = src.read(1)  # 读取第一波段数据
        SAR_bounds = src.bounds  # 获取影像边界
        SAR_transform = src.transform  # 获取影像仿射变换矩阵
        SAR_crs = src.crs  # 获取影像坐标参考系

        # 目标坐标系
        target_crs = "EPSG:4326"

        # 计算转换后的变换矩阵和新尺寸
        transform, width, height = calculate_default_transform(
            SAR_crs, target_crs, src.width, src.height, *src.bounds
        )

        # 创建新的影像数据矩阵
        SAR_image_transformed = np.empty((height, width), dtype=src.dtypes[0])

        # 重采样到目标坐标系
        reproject(
            source=SAR_image,
            destination=SAR_image_transformed,
            src_transform=SAR_transform,
            src_crs=SAR_crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,  # 或者使用其他重采样方法
        )
        
    Selected_ATL03 = ATL03_ALL[(ATL03_ALL['Sort'] == Sort_num) & 
                            (ATL03_ALL['date'] == date) &
                            (ATL03_ALL['subgroup'] == subgroup)].copy()
    Selected_ATL03.rename(columns={'height': 'h'}, inplace=True)
    
    Selected_ATL03_Noise = ATL03_Noise[(ATL03_Noise['Sort'] == Sort_num) &
                            (ATL03_Noise['date'] == date) &
                            (ATL03_Noise['subgroup'] == subgroup)].copy() 
    Selected_ATL03_Noise.rename(columns={'height': 'h'}, inplace=True)

    Selected_ATL06 = ATL06_ALL[(ATL06_ALL['Sort'] == Sort_num) & 
                            (ATL06_ALL['date'] == date) &
                            (ATL06_ALL['subgroup'] == subgroup)].copy()
    Selected_ATL06.rename(columns={'height': 'h'}, inplace=True)

    Selected_ATL08 = ATL08_ALL[(ATL08_ALL['Sort'] == Sort_num) & 
                            (ATL08_ALL['date'] == date) &
                            (ATL08_ALL['subgroup'] == subgroup)].copy()
    Selected_ATL08.rename(columns={'height_centroid': 'h'}, inplace=True)

    Selected_ATL13 = ATL13_ALL[(ATL13_ALL['Sort'] == Sort_num) & 
                            (ATL13_ALL['date'] == date) &
                            (ATL13_ALL['subgroup'] == subgroup)].copy()
    Selected_ATL13.rename(columns={'height_surface': 'h'}, inplace=True)
    
    if len(Selected_ATL03) < 5:
        print(f'{Sort_num}_{date}_{subgroup[0:4]}_{n} ATL03 数据点数量不足')
        continue

    plot_and_get_line(SAR_image_transformed, transform, selected_geometries,selected_area,
                        Selected_ATL03, Selected_ATL03_Noise, 
                        Selected_ATL06,Selected_ATL08, Selected_ATL13, 
                        Sort_num, date, subgroup, n,save_svg=True)
           