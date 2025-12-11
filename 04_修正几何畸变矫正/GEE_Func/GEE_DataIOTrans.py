import ee,os,math,sys
import numpy as np
import geemap
from tqdm import trange
from PackageDeepLearn.utils.Statistical_Methods import Cal_HistBoundary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .GEEMath import get_histogram, get_minmax, get_meanStd, calculate_iou
import pandas as pd
import geopandas as gpd

try:
    from osgeo import gdal
except:
    print('GEE_DataIOTrans not support gdal')
    
class DataTrans(object):
    @staticmethod
    def Open_close(img, radius=10):
        '''
        开闭运算
        '''
        uniformKernel = ee.Kernel.square(**{'radius': radius, 'units': 'meters'})
        min = img.reduceNeighborhood(**{'reducer': ee.Reducer.min(), 'kernel': uniformKernel})
        Openning = min.reduceNeighborhood(**{'reducer': ee.Reducer.max(), 'kernel': uniformKernel})
        max = Openning.reduceNeighborhood(**{'reducer': ee.Reducer.max(), 'kernel': uniformKernel})
        Closing = max.reduceNeighborhood(**{'reducer': ee.Reducer.min(), 'kernel': uniformKernel})
        return Closing
    @staticmethod
    def minmax_norm(Image: ee.Image, Bands, region, scale: int = 10, withbound=False):
        # proj = Image.projection()
        for i, eachName in enumerate(Bands):
            cal_band = Image.select(eachName)
            if withbound:
                histogram = get_histogram(Image, region, scale, histNum=1000).getInfo()
                bin_centers, counts = (np.array(histogram['bucketMeans']), np.array(histogram['histogram']))
                HistBound = Cal_HistBoundary(counts, y=100)
                Min = bin_centers[HistBound['indexFront']]
                Max = bin_centers[HistBound['indexBack']]
                Image = Image.where(Image.lt(Min), Min)
                Image = Image.where(Image.gt(Max), Max)

                nominator = cal_band.subtract(ee.Number(Min))
                denominator = ee.Number(Max).subtract(ee.Number(Min))
            else:
                minmax = get_minmax(cal_band, scale=scale)
                nominator = cal_band.subtract(ee.Number(minmax.get('min')))
                denominator = ee.Number(minmax.get('max')).subtract(ee.Number(minmax.get('min')))

            if i == 0:
                result = nominator.divide(denominator)  # .reproject(proj)
            else:
                result = result.addBands(nominator.divide(denominator))  # .reproject(proj)

        return result

    @staticmethod
    def meanStd_norm(Image: ee.Image, Bands, scale: int = 10):
        '''Z-Score标准化'''
        for i, eachName in enumerate(Bands):
            cal_band = Image.select(eachName)
            meanStd = get_meanStd(cal_band, scale=scale)
            if i == 0:
                result = cal_band.subtract(ee.Number(meanStd.get('mean'))).divide(ee.Number(meanStd.get('std')))
            else:
                result = result.addBands(cal_band.subtract(ee.Number(meanStd.get('mean')))
                                         .divide(ee.Number(meanStd.get('std'))))
        return result

    @staticmethod
    def rm_nodata(col, AOI,bandName='VV',scale=10,maxPixels=1e12):
        # 将遮罩外的元素置-99,默认的遮罩为Nodata区域，并统计Nodata的数量
        # 使用s1_ascending.filter(ee.Filter.eq('numNodata', 0))滤波
        allNone_num = col.select(bandName).unmask(-99).eq(-99).reduceRegion(
            **{
                'geometry': AOI,
                'reducer': ee.Reducer.sum(),
                'scale': scale,
                'maxPixels': maxPixels,
                'bestEffort': True,
            }).get(bandName)
        return col.set({'numNodata': allNone_num})
    
    @staticmethod
    def cal_minmax(image,AOI,bandName='angle',scale=10):
        Obj = get_minmax(image.select(bandName),AOI=AOI,scale=scale)
        return image.set({'min':Obj.get('min'),'max':Obj.get('max')})

    @staticmethod
    def Eq_pixels(x):
        '''将图像像素数量与经纬度等匹配'''
        return ee.Image.constant(0).where(x, x).updateMask(x.mask())

class img_sharp(object):
    '''图像锐化'''
    @staticmethod
    def DoG(Img:ee.Image,fat_radius:int=3,fat_sigma:float=1.,
            skinny_radius:int=3,skinny_sigma:float=0.5):
        '''Difference of Gaussians (DoG)'''
        # Create the Difference of Gaussians (DoG) kernel
        fat = ee.Kernel.gaussian(radius=fat_radius, sigma=fat_sigma, units='pixels')
        skinny = ee.Kernel.gaussian(radius=skinny_radius, sigma=skinny_sigma, units='pixels')

        # Convolve the image with the Gaussian kernels
        convolved_fat = Img.convolve(fat)
        convolved_skinny = Img.convolve(skinny)

        return convolved_fat.subtract(convolved_skinny)

    @staticmethod
    def Laplacian(Img:ee.Image,normalize:bool=True):
        '''Laplacian'''
        # Create the Laplacian kernel
        kernel = ee.Kernel.laplacian8(normalize=normalize)

        # Convolve the image with the Laplacian kernel
        return Img.convolve(kernel)

    @staticmethod
    def Laplacian_of_Gaussian(Img:ee.Image,radius:int=3,sigma:float=1.):
        '''Laplacian of Gaussian'''

class BandTrans(object):

    @staticmethod
    def delBands(Image: ee.Image, *BandsNames):
        '''删除ee.Image中的Bands'''
        Bands_ = Image.bandNames()
        for each in BandsNames:
            Bands_ = Bands_.remove(each)
        return Image.select(Bands_)

    @staticmethod
    def replaceBands(Image1: ee.Image, Image2: ee.Image):
        '''
        Image1: 需要替换的ee.Image
        Image2: 替换的ee.Image
        '''
        Bands1_ = Image1.bandNames()
        Bands2_ = Image2.bandNames()
        return Image1.select(Bands1_.removeAll(Bands2_)).addBands(Image2)

    @staticmethod
    def rename_band(img_path, new_names: list, rewrite=False):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File {img_path} not found.")
        
        if not os.access(img_path, os.W_OK):
            raise PermissionError(f"No write permission for file {img_path}.")

        ds = gdal.Open(img_path, gdal.GA_Update)
        if ds is None:
            raise FileNotFoundError(f"Unable to open file {img_path}.")

        band_count = ds.RasterCount
        if band_count != len(new_names):
            raise ValueError('BandNames length does not match the number of bands.')

        try:
            for i in range(band_count):
                ds.GetRasterBand(i + 1).SetDescription(new_names[i])
            ds.FlushCache()  # Ensure all changes are written

            driver = gdal.GetDriverByName('GTiff')
            if rewrite:
                temp_file = img_path + '.tmp'
                dst_ds = driver.CreateCopy(temp_file, ds)
                dst_ds.FlushCache()
                dst_ds = None
                ds = None

                # Replace the original file with the temporary file
                os.replace(temp_file, img_path)
            else:
                DirName = os.path.dirname(img_path)
                BaseName = os.path.basename(img_path).split('.')[0] + '_Copy.' + os.path.basename(img_path).split('.')[1]
                dst_ds = driver.CreateCopy(os.path.join(DirName, BaseName), ds)
                dst_ds.FlushCache()
                dst_ds = None
        except Exception as e:
            raise RuntimeError(f"An error occurred while renaming bands: {e}")
        finally:
            ds = None  # Ensure dataset is closed

    # 将layover和shadow转为RGB
    @staticmethod
    def add_DistorRgbmask(image, columnNames=[['layover', 'Llayover'], ['shadow'], ['Rlayover']], **parms):
        # 新建波段
        r = ee.Image.constant(0).rename(['red'])
        g = ee.Image.constant(0).rename(['green'])
        b = ee.Image.constant(0).rename(['blue'])

        # 定义一个条件函数来更新RGB图层
        def update_band(band, condition, value):
            return ee.Image(ee.Algorithms.If(condition, band.where(value, 128), band))

        # 遍历参数并根据条件更新波段
        for key, value in parms.items():
            condition = value.bandNames().length().gt(0)

            # 使用服务器端条件判断而不是客户端getInfo
            r = update_band(r, condition.And(ee.List(columnNames[0]).contains(key)), value)
            g = update_band(g, condition.And(ee.List(columnNames[1]).contains(key)), value)
            b = update_band(b, condition.And(ee.List(columnNames[2]).contains(key)), value)

        # 合并波段并更新掩码
        return ee.Image.cat([r, g, b]).byte().updateMask(image.mask())

class Vector_process(object):
    @staticmethod
    def clip_AOI(col, AOI):
        return col.clip(AOI)

    @staticmethod
    def cut_geometryGEE(geometry, block_size: float = 0.05):
        '''
        block_size 定义方块大小(地理坐标系度),0.01约等于1km
        '''
        # 计算边界
        bounds = ee.List(ee.List(geometry.bounds().coordinates()).get(0))

        # 计算geometry的宽度和高度
        width = ee.Number(ee.List(bounds.get(2)).get(0)).subtract(ee.Number(ee.List(bounds.get(0)).get(0)))
        height = ee.Number(ee.List(bounds.get(2)).get(1)).subtract(ee.Number(ee.List(bounds.get(0)).get(1)))

        # 计算行和列的数量
        num_rows = height.divide(block_size).ceil().getInfo()
        num_cols = width.divide(block_size).ceil().getInfo()

        # 定义一个函数，用于生成512x512的方块
        def create_blocks(row, col):
            x_min = ee.Number(ee.List(bounds.get(0)).get(0)).add(col.multiply(block_size))
            y_min = ee.Number(ee.List(bounds.get(0)).get(1)).add(row.multiply(block_size))
            x_max = x_min.add(block_size)
            y_max = y_min.add(block_size)
            return ee.Geometry.Rectangle([x_min, y_min, x_max, y_max])

        # 生成方块列表
        block_list = []
        for row in trange(num_rows):
            for col in trange(num_cols):
                block = create_blocks(ee.Number(row), ee.Number(col))
                block_list.append(block)
        return block_list

    @staticmethod
    def cut_geometry(geometry, block_size: float = 0.05):
        block_size = 0.05
        bounds = geometry.bounds().coordinates().getInfo()[0]

        width = bounds[2][0] - bounds[0][0]
        height = bounds[2][1] - bounds[0][1]
        num_rows = math.ceil(height / block_size)
        num_cols = math.ceil(width / block_size)

        def create_blocks(row, col):
            x_min = bounds[0][0] + col * block_size
            y_min = bounds[0][1] + row * block_size
            x_max = x_min + block_size
            y_max = y_min + block_size
            return ee.Geometry.Rectangle([x_min, y_min, x_max, y_max])

        # 生成方块列表
        block_list = []
        for row in trange(num_rows):
            for col in trange(num_cols):
                block = create_blocks(row, col)
                block_list.append(block)
        return block_list
    
    def split_rectangle_into_grid(AOI, rows, cols):
        """整齐分割矩形边界，比geemap.fishnet可靠"""

        # 获取AOI的边界坐标
        coords = ee.List(AOI.coordinates().get(0))
        x_min = ee.Number(ee.List(coords.get(0)).get(0))
        y_min = ee.Number(ee.List(coords.get(0)).get(1))
        x_max = ee.Number(ee.List(coords.get(2)).get(0))
        y_max = ee.Number(ee.List(coords.get(2)).get(1))
        
        # 计算每个小矩形的宽度和高度
        width = x_max.subtract(x_min).divide(cols)
        height = y_max.subtract(y_min).divide(rows)
        
        # 定义一个函数来创建小矩形
        def create_rectangle(row, col):
            x1 = x_min.add(col.multiply(width))
            y1 = y_min.add(row.multiply(height))
            x2 = x1.add(width)
            y2 = y1.add(height)
            return ee.Geometry.Rectangle([x1, y1, x2, y2])
        
        # 生成小矩形列表
        grid_list = []
        for row in range(rows):
            for col in range(cols):
                rectangle = create_rectangle(ee.Number(row), ee.Number(col))
                grid_list.append(rectangle)
        
        return ee.List(grid_list)

class DataIO(object):
    @staticmethod
    def Geemap_export(fileDirname, input,region=None, scale=10,rename_image=True, keep_zip=True):
        '''
        fileDirname , if collection : xxx
        fileDirname , if collection : xxx.tif
        fileDirname , if vector : xxx.shp

        collection : ee.ImageCollection
        image : ee.Image
        vector : ee.FeatureCollection
        '''

        if isinstance(input, ee.ImageCollection):
            # 这里导出时候使用region设置AOI，否则可能因为坐标系问题(未确定)，出现黑边问题,AOI需要为长方形
            geemap.ee_export_image_collection(input,
                                                out_dir=os.path.dirname(fileDirname),
                                                format="ZIPPED_GEO_TIFF", region=region, scale=scale)
        elif isinstance(input, ee.Image):
            if os.path.exists(fileDirname):
                print('File already exists:{}'.format(fileDirname));pass
            else:
                geemap.ee_export_image(input,
                                        filename=fileDirname,
                                        scale=scale, region=region, file_per_band=False, timeout=300)
                if rename_image:
                    print('change image bandNames')
                    BandTrans.rename_band(fileDirname, new_names=input.bandNames().getInfo(), rewrite=True)
        elif isinstance(input,ee.FeatureCollection) or isinstance(input,ee.Feature):
            if os.path.exists(fileDirname):
                print('File already exists:{}'.format(fileDirname));pass
            else:
                geemap.ee_export_vector(input, fileDirname, selectors=None, verbose=True, 
                                        keep_zip=keep_zip, timeout=300,proxies=None)
        else:
            print('Erro:collection && image must have one False')

    # 从ee.ImageCollection中任意选取ee.Image
    def ImageFromCollection(ImageCollection: ee.ImageCollection, i):
        return ee.Image(ImageCollection.toList(ImageCollection.size()).get(i))

    try:
        import rasterio as rio
        from rasterio.warp import calculate_default_transform
        # 将本地图片导入GEE，并赋予坐标
        def getTransform(tif_path, dst_crs="EPSG:4326"):
            with rio.open(tif_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
            # # 搭配numpy_to_ee可以将图片转为numpy导入ee.Image,但坐标似乎还是有差距
            # img = geemap.numpy_to_ee(p.flip(np.swapaxes(arr[0], 0, 1),axis=1),  crs='EPSG:4326',
            #               transform=[9.622423604352333e-05, 0.0, 91.18529632770264,0.0, -9.622423604352333e-05, 29.499564192382188], band_names='a')
            return transform
    except:
        print('rasterio not import')

class save_parms(object):
    
    @staticmethod
    def save_log(log, mode='gpd', crs='EPSG:4326', logname='log.csv', shapname='log.shp'):
        '''
        mode = 'pd' or 'gpd'
        '''
        if os.path.exists(logname):
            log.drop('geometry', axis=1, inplace=False).to_csv(logname, mode='a', index=False, header=0)
        else:
            log.drop('geometry', axis=1, inplace=False).to_csv(logname, mode='w', index=False)

        if os.path.exists(shapname):
            if mode == 'gpd':
                log.crs = crs
                log.to_file(shapname, driver='ESRI Shapefile', mode='a')
        else:
            if mode == 'gpd':
                log.crs = crs
                log.to_file(shapname, driver='ESRI Shapefile', mode='w')

    # ---------------------------------专用于S1冰湖提取
    @staticmethod
    def write_pd(Union_ex, index, lake_geometry, Img, mode='gpd', Method='SNIC_Kmean', Band=[0, 1, 3], WithOrigin=0,
                 pd_dict=None,
                 Area_real=None, logname='log.csv', shapname='log.shp', calIoU=False, cal_resultArea=False,
                 returnParms=False):

        if cal_resultArea:
            Area_ = Union_ex.area().divide(ee.Number(1000 * 1000)).getInfo()
        else:
            Area_ = False

        if calIoU:
            IoU = calculate_iou(Union_ex, lake_geometry).getInfo()
        else:
            IoU = False

        if mode == 'gpd':
            log = gpd.GeoDataFrame.from_features([Union_ex.getInfo()])
            log = log.assign(**{'Method': Method,
                                'Image': Img,
                                'Band': str(Band),
                                'WithOrigin': WithOrigin,
                                **pd_dict,
                                'Area_pre': [Area_],
                                'Area_real': [Area_real],
                                'IoU': IoU,
                                'index': index})
        else:
            log = pd.DataFrame({'Method': Method,
                                'Image': Img,
                                'Band': str(Band),
                                'WithOrigin': WithOrigin,
                                **pd_dict,
                                'Area_pre': [Area_],
                                'Area_real': [Area_real],
                                'IoU': IoU, },
                               index=[index])

        save_parms.save_log(log, mode=mode, logname=logname, shapname=shapname)
        if returnParms:
            return {'Method': Method,
                    'Image': Img,
                    'Band': str(Band),
                    'WithOrigin': WithOrigin,
                    **pd_dict,
                    'Area_pre': [Area_],
                    'Area_real': [Area_real],
                    'IoU': IoU,
                    'index': index}
