import ee
import geemap
import math
import sys, os
from osgeo import gdal
import asyncio
from GEE_Func.S1_distor_dedicated import load_S1collection, S1_CalDistor, DEM_caculator
from GEE_Func.S2_filter import merge_s2_collection
from GEE_Func.GEE_DataIOTrans import DataTrans, DataIO,Vector_process
from GEE_Func.GEE_CorreterAndFilters import ImageFilter, S1Corrector
from GEE_Func.GEEMath import get_minmax
from GEE_Func.GEE_Tools import Select_imageNum
from functools import partial
import traceback
Eq_pixels = DataTrans.Eq_pixels
from PackageDeepLearn.utils import DataIOTrans
from samgeo import tms_to_geotiff, SamGeo

geemap.set_proxy(port=10809)
# ee.Authenticate()
ee.Initialize()

restrict_Fuse = False  # 图像融合方式
Filter_Angle = 38  # 数字越小越容易导入Layover图像，普遍值为32-45
Nodata_tore = 0  # 感兴趣区域nodata像素数容忍度
NodataTovalue = 0
Origin_scale = 10  # 原始数据分辨率
projScale = 30  # 投影分辨率
box_fromGEE = True  # box是否由GEE获得
BoundBuffer_Add = 300 # 边界缓冲区加值
model = 'volume'  # Slop correction model
DEM = ee.Image("NASA/NASADEM_HGT/001").select('elevation')

years = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']
SETP_Season = ['-02-25', '-05-31', '-09-15', '-11-28', '-02-25']
START_DATE = ee.Date(years[4] + SETP_Season[1])
END_DATE = ee.Date(years[4] + SETP_Season[2])

os.chdir(r'D:\Dataset_and_Demo\DataFused')
# --------------------------------------功能函数
def getS1Corners(image, orbitProperties_pass):
    # 真实方位角(根据整幅影响运算)
    coords = ee.Array(image.geometry().coordinates().get(0)).transpose()
    crdLons = ee.List(coords.toList().get(0))
    crdLats = ee.List(coords.toList().get(1))
    minLon = crdLons.sort().get(0)
    minLat = crdLats.sort().get(0)
    azimuth = (ee.Number(crdLons.get(crdLats.indexOf(minLat))).subtract(minLon).atan2(
        ee.Number(crdLats.get(crdLons.indexOf(minLon))).subtract(minLat))
                .multiply(180.0 / 3.1415926))

    if orbitProperties_pass == 'ASCENDING':
        azimuth = azimuth.add(270.0)
    elif orbitProperties_pass == 'DESCENDING':
        azimuth = azimuth.add(180.0)
    else:
        raise TypeError

    return azimuth


def S1_slope_correction(image, orbitProperties_pass, DEM=DEM, scale=Origin_scale, model=model):
    # 获取投影几何
    geom = image.geometry()
    proj = image.select(0).projection()

    # 计算方位向、距离向以及图像的四个角点、构筑计算辅助线
    Heading = getS1Corners(image, orbitProperties_pass)  # 'ASCENDING' = 348.3   'DESCENDING' = 189.2

    s1_azimuth_across = ee.Number(Heading).subtract(90.0)  # 距离向
    theta_iRad = image.select('angle').multiply(3.1415926 / 180)  # 地面入射角度转为弧度
    phi_iRad = ee.Image.constant(s1_azimuth_across).multiply(3.1415926 / 180)  # 距离向转弧度

    # 计算地形几何信息
    _, _, alpha_rRad, alpha_azRad = DEM_caculator.slop_aspect(DEM, proj, geom, phi_iRad)

    # 根据入射角度修订
    sigma0Pow = ee.Image.constant(10).pow(image.divide(10.0))
    gamma0 = sigma0Pow.divide(theta_iRad.cos())

    slop_correction = S1Corrector.volumetric(model, theta_iRad, alpha_rRad, alpha_azRad, gamma0)
    image_ = (Eq_pixels(slop_correction['gamma0_flatDB'].select('VV_gamma0flat')).rename('VV_gamma0_flatDB')
              .addBands(Eq_pixels(slop_correction['gamma0_flatDB'].select('VH_gamma0flat')).rename('VH_gamma0_flatDB'))
              .addBands(Eq_pixels(image.select('angle')).rename('incAngle'))).reproject(crs=proj)

    # 手动复制时间信息
    time_start = image.get('system:time_start')
    time_end = image.get('system:time_end')
    image_ = image_.set('system:time_start', time_start).set('system:time_end', time_end)
    return image_.copyProperties(image)


def getMask(image, nodataValues: list = []):
    combined_mask = image.mask()  # 获取现有的掩膜
    if len(nodataValues) == 0:
        return combined_mask
    else:
        for nodatavalue in nodataValues:
            value_mask = image.neq(nodatavalue).unmask(0)
            combined_mask = combined_mask.And(value_mask)
        return combined_mask


def export_image_tiles(image, save_path, grid_list,tiles_num, scale):
    tasks = []
    for idx in range(tiles_num):
        tile_path = f"{save_path}_tile_{idx}.tif"
        geemap.ee_export_image(image, filename=tile_path, scale=scale, region=grid_list.get(idx), file_per_band=False, timeout=300)
        tasks.append(tile_path)
    return tasks

def merge_tiles(tile_paths, output_path):
    vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', addAlpha=True)
    vrt = gdal.BuildVRT('/vsimem/temporary.vrt', tile_paths, options=vrt_options)
    gdal.Translate(output_path, vrt)
    vrt = None  # 释放内存

# --------------------------预加载冰湖数据,测试的时候加上Filter_bound
Glacial_lake = ee.FeatureCollection('projects/ee-mrwurenzhe/assets/Glacial_lake/SAR_GLs/2019Gls_SARExt').sort('fid_1')
Glacial_lake_Shrink = ee.FeatureCollection('projects/ee-mrwurenzhe/assets/Glacial_lake/SAR_GLs/2019Gls_SARExt_inbuffer').sort('fid_1')
# --------------------------预加载几何畸变数据
Ascending_DistorFull = ee.Image('projects/ee-mrwurenzhe/assets/SETP_Distor/SETPAscending_Distor')
Descending_DistorFull = ee.Image('projects/ee-mrwurenzhe/assets/SETP_Distor/SETPDescending_Distor')
Ascending_GradFull = ee.Image('projects/ee-mrwurenzhe/assets/SETP_Distor/SETPAscending_Forshortening_Grading_30')
Descending_GradFull = ee.Image('projects/ee-mrwurenzhe/assets/SETP_Distor/SETPDescending_Forshortening_Grading_30')

# 计算geometry、质心点、最小包络矩形
Geo_ext = lambda feature: feature.set({
    'Geo': feature.geometry(),
    'Centroid': feature.geometry().centroid(),
    'Rectangle': feature.geometry().bounds()})

Glacial_lake_C = Glacial_lake.map(Geo_ext)
Num_list = Glacial_lake.size().getInfo()
Glacial_lake_A_GeoList = Glacial_lake.toList(Num_list)
Glacial_lake_C_CentriodList = ee.List(Glacial_lake_C.reduceColumns(ee.Reducer.toList(), ['Centroid']).get('list'))
Glacial_lake_R_RectangleList = ee.List(Glacial_lake_C.reduceColumns(ee.Reducer.toList(), ['Rectangle']).get('list'))
Glacial_lake_Shrink_GeoList = Glacial_lake_Shrink.toList(Num_list)


async def download_data(i, semaphore):
    async with semaphore:
        try:
            GLAOI = ee.Feature(Glacial_lake_A_GeoList.get(i))
            AOI = GLAOI.geometry()
            AOI_Bound = ee.Feature(Glacial_lake_R_RectangleList.get(i)).geometry()
            AOI_Centro = ee.Feature(Glacial_lake_C_CentriodList.get(i)).geometry()
            AOI_area = float(GLAOI.get('Area_pre').getInfo())
            AOI_bufferBounds = AOI_Bound.buffer(distance=math.log(AOI_area + 1, 5) * 1200 + BoundBuffer_Add).bounds()

            s1_col = (ee.ImageCollection("COPERNICUS/S1_GRD")
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filterBounds(AOI)
                    .filterDate(START_DATE, END_DATE))

            s1_col = s1_col.map(partial(DataTrans.rm_nodata, AOI=AOI))
            s1_col = s1_col.map(partial(DataTrans.cal_minmax, AOI=AOI))
            s1_col_Nodata = s1_col.filter(ee.Filter.lte('numNodata', Nodata_tore))
            if s1_col_Nodata.size().getInfo() == 0:
                s1_col_Nodata = s1_col
                print('All s1_col data nodata nums > 0')

            s1_col_Angle = s1_col_Nodata.filter(ee.Filter.gte('min', Filter_Angle))
            if s1_col_Angle.size().getInfo() == 0:
                s1_col_Angle = s1_col_Nodata
                print('All s1_col data Angle < {}'.format(Filter_Angle))

            s1_col = s1_col_Nodata.map(ImageFilter.RefinedLee)
            proj = s1_col.first().select(0).projection()

            s1_ascending_collection = s1_col.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
                                            .map(partial(S1_slope_correction, orbitProperties_pass='ASCENDING'))
            s1_descending_collection = s1_col.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
                                            .map(partial(S1_slope_correction, orbitProperties_pass='DESCENDING'))

            s1_ascending = s1_ascending_collection.mean().reproject(crs=proj).clip(AOI_bufferBounds)
            s1_descending = s1_descending_collection.mean().reproject(crs=proj, scale=Origin_scale).clip(AOI_bufferBounds)

            Ascending_Distor = Ascending_DistorFull
            Descending_Distor = Descending_DistorFull
            A_Mask = getMask(Ascending_Distor, nodataValues=[9])
            D_Mask = getMask(Descending_Distor, nodataValues=[9])
            Ascending_Grad = Ascending_GradFull
            Descending_Grad = Descending_GradFull
            A_Mask = A_Mask.Or(Ascending_Grad.gt(10).unmask(0))
            D_Mask = D_Mask.Or(Descending_Grad.gt(10).unmask(0))
            A_Mask = Eq_pixels(A_Mask).reproject(crs=proj).clip(AOI_bufferBounds)
            D_Mask = Eq_pixels(D_Mask).reproject(crs=proj).clip(AOI_bufferBounds)

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
            save_path = "{}_ADMeanFused.tif".format(f'{i:05d}')
            
            # 尝试直接导出完整图像                  
            try:
                DataIO.Geemap_export(save_path, s1_unit_mean_, region=AOI_bufferBounds, scale=Origin_scale, rename_image=False)
                # 检查文件是否成功下载
                if not os.path.exists(save_path):
                    raise FileNotFoundError(f"{save_path} not found after export")
            except Exception as e:
                print(f'直接导出失败: {e}')
                print('开始分块导出...')
                
                grid_list = Vector_process.split_rectangle_into_grid(AOI_bufferBounds, 3, 3)
                tile_paths = export_image_tiles(s1_unit_mean_, save_path, grid_list,9,Origin_scale)
                save_path = "{}_ADMeanFused_WithTiles.tif".format(f'{i:05d}')
                merge_tiles(tile_paths, save_path)
                
                for tile_path in tile_paths:
                    if os.path.exists(tile_path):
                        os.remove(tile_path)                    
            

            # 确保文件存在
            if not os.path.exists(save_path):
                raise FileNotFoundError(f"{save_path} not found after export")

            imagePath = DataIOTrans.DataIO.TransImage_Values(save_path, transFunc=DataIOTrans.DataTrans.MinMaxBoundaryScaler,
                                                            bandSave=[0, 0, 0], scale=255)

            return imagePath, AOI_Bound

        except Exception as e:
            print(f'错误发生: {e}')
            # 记录错误信息
            with open('log.txt', 'a') as f:
                f.write(f'Wrong index = {i}\n')
                f.write(traceback.format_exc())
                f.write('\n')
            print(f'错误已记录到log.txt: {e}')
            return None, None


async def predict_image(i, imagePath, AOI_Bound, semaphore):
    async with semaphore:
        try:
            sam = SamGeo(
                model_type="vit_h",
                automatic=False,
                sam_kwargs=None,
            )
            sam.set_image(imagePath)

            Output_Path = "{}_ADMeanFused_SAM.tif".format(f'{i:05d}')
            AOI_Bound_Info = AOI_Bound.getInfo()
            if box_fromGEE:
                boxes = [*AOI_Bound_Info['coordinates'][0][0], *AOI_Bound_Info['coordinates'][0][2]]
                sam.predict(boxes=boxes, point_crs="EPSG:4326", output=Output_Path, dtype="uint8")

        except Exception as e:
            print(f'预测时发生错误: {e}')
            # 记录错误信息
            with open('log.txt', 'a') as f:
                f.write(f'Predict error index = {i}\n')
                f.write(traceback.format_exc())
                f.write('\n')
            print(f'预测错误已记录到log.txt: {e}')

async def process_index(i, semaphore):
    print(f"Processing index: {i}")
    imagePath, AOI_Bound = await download_data(i, semaphore)
    if imagePath and AOI_Bound:
        await predict_image(i, imagePath, AOI_Bound, semaphore)

async def main():
    semaphore = asyncio.Semaphore(10)  # 限制同时运行的任务数量
    Cal_list = list(range(Num_list))

    # 过滤掉已存在文件的索引
    Cal_list = [i for i in Cal_list if not ((os.path.exists("{}_ADMeanFused.tif".format(f'{i:05d}')) 
                                            or os.path.exists("{}_ADMeanFused_WithTiles.tif".format(f'{i:05d}')))
                                            and os.path.exists("{}_ADMeanFused_SAM.tif".format(f'{i:05d}')))                                                ]

    tasks = [process_index(i, semaphore) for i in Cal_list]
    await asyncio.gather(*tasks)

# 运行主程序
if __name__ == "__main__":
    asyncio.run(main())
