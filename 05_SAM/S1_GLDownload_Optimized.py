import ee
import geemap
import math
import sys, os
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import traceback
from osgeo import gdal
import numpy as np
import time
from threading import Lock

# 设置环境变量
os.environ["PROJ_LIB"] = r"C:\ProgramData\anaconda3\envs\GEE\Lib\site-packages\pyproj\proj_dir\share\proj"
os.environ.setdefault("PROJ_DATA", os.environ["PROJ_LIB"])

# 确保项目根目录在sys.path中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GEE_Func.S1_distor_dedicated import DEM_caculator
from GEE_Func.GEE_DataIOTrans import DataTrans, DataIO, Vector_process
from GEE_Func.GEE_CorreterAndFilters import ImageFilter, S1Corrector
from GEE_Func.GEE_Tools import S1_Cheker
from PackageDeepLearn.utils import DataIOTrans


Eq_pixels = DataTrans.Eq_pixels

# 配置参数
geemap.set_proxy(port=10809)
ee.Initialize()

# 全局配置
restrict_Fuse = False
Filter_Angle = 38
Nodata_tore = 0
NodataTovalue = 0
Origin_scale = 10
projScale = 30
box_fromGEE = True
BoundBuffer_Add = 300
model = 'volume'
DEM = ee.Image("NASA/NASADEM_HGT/001").select('elevation')
SaveDir = r'E:\SETP_GL'

years = ['2024','2025']
SETP_Season = ['-11-28', '-02-25']

# 并行处理配置
MAX_WORKERS = min(16, mp.cpu_count())  # 限制并发数，避免GEE API限制
BATCH_SIZE = 10  # 批处理大小

# 全局缓存
_cache_lock = Lock()
_geometry_cache = {}
_collection_cache = {}

# 预加载数据（优化：减少重复加载）
print("正在预加载数据...")
# Glacial_lake = ee.FeatureCollection('projects/ee-mrwurenzhe/assets/Glacial_lake/SAR_GLs/GL_replenish')
Glacial_lake = ee.FeatureCollection('projects/ee-mrwurenzhe/assets/Glacial_lake/2023_05_31_to_2023_09_15_SpatialJoin').map(
    lambda feature: feature.set('ID', ee.Number.parse(feature.get('ID')))
).sort('ID')

# 几何畸变数据
Ascending_DistorFull = ee.Image('projects/ee-mrwurenzhe/assets/SETP_Distor/SETPAscending_Distor')
Descending_DistorFull = ee.Image('projects/ee-mrwurenzhe/assets/SETP_Distor/SETPDescending_Distor')
Ascending_GradFull = ee.Image('projects/ee-mrwurenzhe/assets/SETP_Distor/SETPAscending_Forshortening_Grading_30')
Descending_GradFull = ee.Image('projects/ee-mrwurenzhe/assets/SETP_Distor/SETPDescending_Forshortening_Grading_30')

# 预计算几何信息（优化：批量处理几何计算）
def precompute_geometries():
    """预计算所有冰湖的几何信息，减少运行时计算"""
    global _geometry_cache
    
    if _geometry_cache:
        return _geometry_cache
    
    print("预计算几何信息...")
    
    # 批量获取几何信息
    Geo_ext = lambda feature: feature.set({
        'Geo': feature.geometry(),
        'Centroid': feature.geometry().centroid()
    })
    
    Glacial_lake_C = Glacial_lake.map(Geo_ext)
    
    # 一次性获取所有需要的信息
    try:
        num_list = Glacial_lake.size().getInfo()
        glacial_lake_list = Glacial_lake.toList(num_list).getInfo()
        centroid_list = Glacial_lake_C.reduceColumns(ee.Reducer.toList(), ['Centroid']).get('list').getInfo()
        
        _geometry_cache = {
            'num_list': num_list,
            'glacial_lake_list': glacial_lake_list,
            'centroid_list': centroid_list
        }
        
        print(f"预计算完成，共{num_list}个冰湖")
        return _geometry_cache
        
    except Exception as e:
        print(f"预计算几何信息失败: {e}")
        return None

def getS1Corners(image, orbitProperties_pass):
    """计算S1影像的方位角"""
    coords = ee.Array(image.geometry().coordinates().get(0)).transpose()
    crdLons = ee.List(coords.toList().get(0))
    crdLats = ee.List(coords.toList().get(1))
    minLon = crdLons.sort().get(0)
    minLat = crdLats.sort().get(0)
    azimuth = (ee.Number(crdLons.get(crdLats.indexOf(minLat))).subtract(minLon).atan2(
        ee.Number(crdLats.get(crdLons.indexOf(minLon))).subtract(minLat))
                .multiply(180.0 / 3.131415926))

    if orbitProperties_pass == 'ASCENDING':
        azimuth = azimuth.add(270.0)
    elif orbitProperties_pass == 'DESCENDING':
        azimuth = azimuth.add(180.0)
    else:
        raise TypeError
    
    return azimuth

def S1_slope_correction(image, orbitProperties_pass, DEM=DEM, scale=Origin_scale, model=model):
    """S1坡度校正"""
    geom = image.geometry()
    proj = image.select(0).projection()

    Heading = getS1Corners(image, orbitProperties_pass)
    s1_azimuth_across = ee.Number(Heading).subtract(90.0)
    theta_iRad = image.select('angle').multiply(3.1415926 / 180)
    phi_iRad = ee.Image.constant(s1_azimuth_across).multiply(3.1415926 / 180)

    _, _, alpha_rRad, alpha_azRad = DEM_caculator.slop_aspect(DEM, proj, geom, phi_iRad)

    sigma0Pow = ee.Image.constant(10).pow(image.divide(10.0))
    gamma0 = sigma0Pow.divide(theta_iRad.cos())

    slop_correction = S1Corrector.volumetric(model, theta_iRad, alpha_rRad, alpha_azRad, gamma0)
    image_ = (Eq_pixels(slop_correction['gamma0_flatDB'].select('VV_gamma0flat')).rename('VV_gamma0_flatDB')
            .addBands(Eq_pixels(slop_correction['gamma0_flatDB'].select('VH_gamma0flat')).rename('VH_gamma0_flatDB'))
            .addBands(Eq_pixels(image.select('angle')).rename('incAngle'))).reproject(crs=proj)
    
    time_start = image.get('system:time_start')
    time_end = image.get('system:time_end')
    image_ = image_.set('system:time_start', time_start).set('system:time_end', time_end)
    return image_.copyProperties(image)

def getMask(image, nodataValues: list = []):
    """获取掩膜"""
    combined_mask = image.mask()
    if len(nodataValues) == 0:
        return combined_mask
    else:
        for nodatavalue in nodataValues:
            value_mask = image.neq(nodatavalue).unmask(0) 
            combined_mask = combined_mask.And(value_mask)  
        return combined_mask

def export_image_async(image, save_path, region, scale, timeout=600):
    """异步导出图像"""
    try:
        # 使用geemap的异步导出功能
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=os.path.basename(save_path).replace('.tif', ''),
            folder='GEE_Export',
            fileNamePrefix=os.path.basename(save_path).replace('.tif', ''),
            region=region,
            scale=scale,
            maxPixels=1e13
        )
        task.start()
        
        # 等待任务完成
        start_time = time.time()
        while task.active() and (time.time() - start_time) < timeout:
            time.sleep(10)
            
        if task.active():
            task.cancel()
            return False, "导出超时"
            
        state = task.status()['state']
        if state == 'COMPLETED':
            return True, "导出成功"
        else:
            return False, f"导出失败: {state}"
            
    except Exception as e:
        return False, f"导出异常: {str(e)}"

def download_data_optimized(i, START_DATE, END_DATE, output_folder, geometry_cache):
    """优化的数据下载函数"""
    try:
        save_path = os.path.join(output_folder, f"{i:05d}_ADMeanFused.tif")
        save_path2 = os.path.join(output_folder, f"{i:05d}_ADMeanFused_WithTiles.tif")
        
        # 检查文件是否已存在
        if os.path.exists(save_path) or os.path.exists(save_path2):
            print(f"文件已存在，跳过索引 {i}")
            return save_path if os.path.exists(save_path) else save_path2, None
        
        # 从缓存获取几何信息
        glacial_lake_info = geometry_cache['glacial_lake_list'][i]
        AOI = ee.Geometry(glacial_lake_info['geometry'])
        AOI_area_raw = glacial_lake_info['properties'].get('Area_pre', 1000)
        
        # 确保AOI_area是数值类型
        try:
            AOI_area = float(AOI_area_raw) if AOI_area_raw is not None else 1000.0
        except (ValueError, TypeError):
            AOI_area = 1000.0  # 默认值
        
        # 计算缓冲区 - 直接使用AOI的bounds而不是预计算的rectangle
        AOI_Bound = AOI.bounds()
        buffer_distance = math.log(AOI_area + 1, 5) * 1200 + BoundBuffer_Add
        AOI_bufferBounds = AOI_Bound.buffer(distance=buffer_distance).bounds()

        # S1数据收集（优化：减少重复过滤）
        s1_col = (ee.ImageCollection("COPERNICUS/S1_GRD")
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
                  .filterBounds(AOI)
                  .filterDate(START_DATE, END_DATE)
                  .map(partial(DataTrans.rm_nodata, AOI=AOI))
                  .map(partial(DataTrans.cal_minmax, AOI=AOI))
                  )
                  # .map(lambda x: x.updateMask(x.gte(-30)))

        # 获取投影信息
        proj = s1_col.first().select(0).projection()
        
        # 分离升降轨
        s1_a_col = s1_col.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
        s1_d_col = s1_col.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        
        # 优化：批量检查集合大小
        collection_sizes = ee.Dictionary({
            'ascending': s1_a_col.size(),
            'descending': s1_d_col.size()
        }).getInfo()
        
        if collection_sizes['ascending'] == 0 or collection_sizes['descending'] == 0:
            print(f'索引 {i}: 缺少升轨或降轨数据')
            # 使用所有可用数据
            Origin = s1_col.map(S1_Cheker.CheckDuplicateBands).map(ImageFilter.RefinedLee)\
                         .mean().reproject(crs=proj).clip(AOI_bufferBounds)
            s1_unit_mean_ = Origin
        else:
            # 数据过滤和处理
            s1_a_col_filtered = s1_a_col.filter(ee.Filter.lte('numNodata', Nodata_tore))\
                                       .filter(ee.Filter.gte('min', Filter_Angle))
            s1_d_col_filtered = s1_d_col.filter(ee.Filter.lte('numNodata', Nodata_tore))\
                                       .filter(ee.Filter.gte('min', Filter_Angle))
            
            # 如果过滤后没有数据，使用原始数据
            if s1_a_col_filtered.size().getInfo() == 0:
                s1_a_col_filtered = s1_a_col
            if s1_d_col_filtered.size().getInfo() == 0:
                s1_d_col_filtered = s1_d_col

            # 处理升降轨数据
            s1_ascending_collection = s1_a_col_filtered.map(S1_Cheker.CheckDuplicateBands)\
                                                      .map(ImageFilter.RefinedLee)\
                                                      .map(partial(S1_slope_correction, orbitProperties_pass='ASCENDING'))
            s1_descending_collection = s1_d_col_filtered.map(S1_Cheker.CheckDuplicateBands)\
                                                       .map(ImageFilter.RefinedLee)\
                                                       .map(partial(S1_slope_correction, orbitProperties_pass='DESCENDING'))

            s1_ascending = s1_ascending_collection.mean().reproject(crs=proj).clip(AOI_bufferBounds)
            s1_descending = s1_descending_collection.mean().reproject(crs=proj).clip(AOI_bufferBounds)

            # 应用几何畸变掩膜
            A_Mask = getMask(Ascending_DistorFull, nodataValues=[9])
            D_Mask = getMask(Descending_DistorFull, nodataValues=[9])
            A_Mask = A_Mask.Or(Ascending_GradFull.gt(10).unmask(0))
            D_Mask = D_Mask.Or(Descending_GradFull.gt(10).unmask(0))
            A_Mask = Eq_pixels(A_Mask).reproject(crs=proj).clip(AOI_bufferBounds)
            D_Mask = Eq_pixels(D_Mask).reproject(crs=proj).clip(AOI_bufferBounds)
            
            # 融合升降轨数据
            if restrict_Fuse:
                s1_ascending_ = s1_ascending.updateMask(A_Mask.Not())
                s1_descending_ = s1_descending.updateMask(D_Mask.Not())
                Combin_AD = ee.ImageCollection([s1_ascending_, s1_descending_])
                s1_unit_mean_ = Combin_AD.mean().reproject(crs=proj).clip(AOI_bufferBounds)
            else:
                s1_ascending_ = ee.Image(s1_ascending.where(A_Mask, s1_descending))
                s1_descending_ = ee.Image(s1_descending.where(D_Mask, s1_ascending))
                Combin_AD = ee.ImageCollection([s1_ascending_, s1_descending_])
                s1_unit_mean_ = Combin_AD.mean().reproject(crs=proj).clip(AOI_bufferBounds)

            s1_unit_mean_ = s1_unit_mean_.unmask(NodataTovalue)
        
        # 导出图像
        try:
            DataIO.Geemap_export(save_path, s1_unit_mean_, region=AOI_bufferBounds, 
                                scale=Origin_scale, rename_image=False)
            if not os.path.exists(save_path):
                raise FileNotFoundError(f"{save_path} not found after export")
        except Exception as e:
            print(f'索引 {i} 直接导出失败: {e}，尝试分块导出...')
            # 分块导出
            grid_list = Vector_process.split_rectangle_into_grid(AOI_bufferBounds, 3, 3)
            tile_paths = []
            for idx in range(9):
                tile_path = f"{save_path}_tile_{idx}.tif"
                if not os.path.exists(tile_path):
                    geemap.ee_export_image(s1_unit_mean_, filename=tile_path, scale=Origin_scale, 
                                         region=grid_list.get(idx), file_per_band=False, timeout=300)
                tile_paths.append(tile_path)
            
            # 合并分块
            save_path = save_path2
            vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', addAlpha=True)
            vrt = gdal.BuildVRT('/vsimem/temporary.vrt', tile_paths, options=vrt_options)
            gdal.Translate(save_path, vrt)
            vrt = None
            
            # 清理临时文件
            for tile_path in tile_paths:
                if os.path.exists(tile_path):
                    os.remove(tile_path)

        if not os.path.exists(save_path):
            raise FileNotFoundError(f"{save_path} not found after export")

        # 图像后处理
        imagePath = DataIOTrans.DataIO.TransImage_Values(
            save_path, 
            transFunc=DataIOTrans.DataTrans.MinMaxBoundaryScaler,
            bandSave=[0, 0, 0], 
            scale=255
        )

        # 保存AOI信息
        # aoi_info_path = save_path.replace('.tif', '_AOI.json')
        # with open(aoi_info_path, 'w') as f:
        #     json.dump(AOI_Bound.getInfo(), f)

        print(f"索引 {i} 处理完成: {imagePath}")
        return imagePath, AOI_Bound

    except Exception as e:
        error_msg = f'索引 {i} 错误: {e}\n{traceback.format_exc()}'
        print(error_msg)
        with open(os.path.join(output_folder, 'log.txt'), 'a', encoding='utf-8') as f:
            f.write(f'Wrong index = {i}\n{error_msg}\n')
        return None, None

def process_batch_parallel(indices, START_DATE, END_DATE, output_folder, geometry_cache):
    """并行处理一批数据"""
    results = []
    
    # 使用线程池进行并行处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(download_data_optimized, i, START_DATE, END_DATE, output_folder, geometry_cache): i 
            for i in indices
        }
        
        # 收集结果
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results.append((index, result))
                print(f"完成处理索引 {index}")
            except Exception as e:
                print(f"索引 {index} 处理失败: {e}")
                results.append((index, (None, None)))
    
    return results

def main_optimized():
    """优化的主函数"""
    print("开始优化版本的数据下载...")
    
    # 预计算几何信息
    geometry_cache = precompute_geometries()
    if not geometry_cache:
        print("预计算失败，退出程序")
        return
    
    total_lakes = geometry_cache['num_list']
    print(f"总共需要处理 {total_lakes} 个冰湖")
    
    for year in years:
        for season_index in range(len(SETP_Season) - 1):
            START_DATE = ee.Date(year + SETP_Season[season_index])
            
            # 处理跨年季节
            if SETP_Season[season_index] == '-11-28' and SETP_Season[season_index + 1] == '-02-25':
                next_year = str(int(year) + 1)
                END_DATE = ee.Date(next_year + SETP_Season[season_index + 1])
            else:
                END_DATE = ee.Date(year + SETP_Season[season_index + 1])
            
            print(f"处理时间段: {START_DATE.format('YYYY-MM-dd').getInfo()} 到 {END_DATE.format('YYYY-MM-dd').getInfo()}")
            
            # 创建输出文件夹
            output_folder = os.path.join(SaveDir, f"{START_DATE.format('YYYY-MM-dd').getInfo()}_to_{END_DATE.format('YYYY-MM-dd').getInfo()}")
            os.makedirs(output_folder, exist_ok=True)
            
            # 获取需要处理的索引列表（跳过已存在的文件）
            all_indices = list(range(total_lakes))
            remaining_indices = []
            
            for i in all_indices:
                save_path1 = os.path.join(output_folder, f"{i:05d}_ADMeanFused.tif")
                save_path2 = os.path.join(output_folder, f"{i:05d}_ADMeanFused_WithTiles.tif")
                if not (os.path.exists(save_path1) or os.path.exists(save_path2)):
                    remaining_indices.append(i)
            
            print(f"需要处理 {len(remaining_indices)} 个冰湖（跳过 {total_lakes - len(remaining_indices)} 个已存在文件）")
            
            if not remaining_indices:
                print("所有文件已存在，跳过此时间段")
                continue
            
            # 分批并行处理
            total_batches = (len(remaining_indices) + BATCH_SIZE - 1) // BATCH_SIZE
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(remaining_indices))
                batch_indices = remaining_indices[start_idx:end_idx]
                
                print(f"处理批次 {batch_idx + 1}/{total_batches}，索引 {batch_indices[0]} 到 {batch_indices[-1]}")
                
                # 并行处理当前批次
                batch_results = process_batch_parallel(batch_indices, START_DATE, END_DATE, output_folder, geometry_cache)
                
                # 统计结果
                successful = sum(1 for _, (path, _) in batch_results if path is not None)
                print(f"批次 {batch_idx + 1} 完成: {successful}/{len(batch_indices)} 成功")
                
                # 短暂休息，避免API限制
                if batch_idx < total_batches - 1:
                    time.sleep(2)
    
    print("所有数据处理完成！")

if __name__ == "__main__":
    main_optimized()