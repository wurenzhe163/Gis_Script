import os
import ee
import geemap
from Correct_filter import *
import copy
from tqdm import tqdm, trange
from New_Correct import *
from Correct_filter import volumetric_model_SCF

def make_dir(path):
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
    return path

def delList(L):
    """"
    删除重复元素
    """
    L1 = []
    for i in L:
        if i not in L1:
            L1.append(i)
    return L1

def Open_close(img,radius=10):
    '''
    开闭运算
    '''
    uniformKernel = ee.Kernel.square(**{ 'radius': radius, 'units': 'meters'})
    min = img.reduceNeighborhood(**{'reducer': ee.Reducer.min(), 'kernel': uniformKernel })
    Openning = min.reduceNeighborhood(**{'reducer': ee.Reducer.max(), 'kernel': uniformKernel })
    max = Openning.reduceNeighborhood(**{'reducer': ee.Reducer.max(), 'kernel': uniformKernel })
    Closing = max.reduceNeighborhood(**{'reducer': ee.Reducer.min(), 'kernel': uniformKernel })
    return Closing

def calculate_iou(geometry1, geometry2):
    intersection = geometry1.intersection(geometry2)
    union = geometry1.union(geometry2)
    intersection_area = intersection.area()
    union_area = union.area()
    return intersection_area.divide(union_area)

#--------------------Norm----------------------
def get_minmax(Image:ee.Image,scale:int=10):
    '''Image 只能包含有一个波段，否则不能重命名'''
    Obj = Image.reduceRegion(reducer=ee.Reducer.minMax(),geometry=Image.geometry(),scale=scale,bestEffort=True)
    return Obj.rename(**{'from': Obj.keys(), 'to': ['max', 'min']})

def get_meanStd(Image:ee.Image,scale:int=10):
    '''Image 只能包含有一个波段，否则不能重命名'''
    Obj = Image.reduceRegion(reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                       geometry=Image.geometry(), scale=scale, bestEffort=True)
    return Obj.rename(**{'from': Obj.keys(), 'to': ['mean', 'std']})

def minmax_norm(Image:ee.Image,Bands,scale:int=10):
    for i,eachName in enumerate(Bands):
        cal_band = Image.select(eachName)
        minmax = get_minmax(cal_band, scale=scale)
        nominator = cal_band.subtract(ee.Number(minmax.get('min')))
        denominator = ee.Number(minmax.get('max')).subtract(ee.Number(minmax.get('min')))
        if i == 0:
            result = nominator.divide(denominator)
        else:
            result = result.addBands(nominator.divide(denominator))
    return result

def meanStd_norm(Image:ee.Image,Bands,scale:int=10):
    '''Z-Score标准化'''
    for i,eachName in enumerate(Bands):
        cal_band = Image.select(eachName)
        meanStd = get_meanStd(cal_band,scale=scale)
        if i == 0:
            result = cal_band.subtract(ee.Number(meanStd.get('mean'))).divide(ee.Number(meanStd.get('std')))
        else:
            result = result.addBands(cal_band.subtract(ee.Number(meanStd.get('mean')))
                                     .divide(ee.Number(meanStd.get('std'))))
    return result

def get_histogram(Image:ee.Image,region,scale,histNum=1000):
    histogram0 = Image.reduceRegion(
        reducer=ee.Reducer.histogram(histNum),
        geometry=region,
        scale=scale,
        maxPixels=1e12,
        bestEffort=True)#自动平衡scale减少计算量
    histogram = histogram0.get(Image.bandNames().get(0))
    return histogram

#---------------------------------计算IoU
def calculate_iou(geometry1, geometry2):
    intersection = geometry1.intersection(geometry2)
    union = geometry1.union(geometry2)
    intersection_area = intersection.area()
    union_area = union.area()
    return intersection_area.divide(union_area)

# --------------------------------直方图匹配
def histogramMatching(sourceImg, targetImg,AOI,source_bandsNames,target_bandsNames,Histscale=30,maxBuckets=256):
    '''
    直方图匹配
    :param sourceImg: 源影像
    :param targetImg: 目标影像
    :param AOI: 匹配区域
    :param Histscale: 直方图匹配的分辨率
    :param maxBuckets: 直方图匹配的最大桶数
    :return: 匹配后的源图像
    '''
    def lookup(sourceHist, targetHist):
        # 第一列数据是原始值，第二列是统计的累积数量
        sourceValues = sourceHist.slice(1, 0, 1).project([0])
        sourceCounts = sourceHist.slice(1, 1, 2).project([0])
        sourceCounts = sourceCounts.divide(sourceCounts.get([-1]))

        targetValues = targetHist.slice(1, 0, 1).project([0])
        targetCounts = targetHist.slice(1, 1, 2).project([0])
        targetCounts = targetCounts.divide(targetCounts.get([-1]))

        # 遍历原始数据值，查找大于这个值的最大索引，然后返回对应的数据
        def _n(n):
            index = targetCounts.gte(n).argmax()
            return targetValues.get(index)

        yValues = sourceCounts.toList().map(_n)
        return {'x': sourceValues.toList(), 'y': yValues}

    source_bandsNames = source_bandsNames
    target_bandsNames = target_bandsNames
    assert len(source_bandsNames) == len(target_bandsNames), 'source and target image must have the same number of bands'

    args = {
        'reducer': ee.Reducer.autoHistogram(**{'maxBuckets': maxBuckets, 'cumulative': True}),
        'geometry': AOI,
        'scale': Histscale,
        'maxPixels': 1e13,
        'tileScale':16
    }
    source = sourceImg.reduceRegion(**args)
    target = targetImg.updateMask(sourceImg.mask()).reduceRegion(**args)

    Copy_sourceImg = []
    for band_source,band_target in zip(source_bandsNames,target_bandsNames):
        Lookup = lookup(source.getArray(band_source), target.getArray(band_target))
        Copy_sourceImg.append(sourceImg.select([band_source]).interpolate(**Lookup))
    return ee.Image.cat(Copy_sourceImg)

# --------------------------------删除波段
def delBands(Image:ee.Image,*BandsNames):
    '''删除ee.Image中的Bands'''
    Bands_ = Image.bandNames()
    for each in BandsNames:
        Bands_ = Bands_.remove(each)
    return Image.select(Bands_)

# ---------------------------------替换波段
def replaceBands(Image1:ee.Image,Image2:ee.Image):
    '''
    Image1: 需要替换的ee.Image
    Image2: 替换的ee.Image
    '''
    Bands1_ = Image1.bandNames()
    Bands2_ = Image2.bandNames()
    return Image1.select(Bands1_.removeAll(Bands2_)).addBands(Image2)

def clip_AOI(col, AOI): return col.clip(AOI)

def cut_geometryGEE(geometry,block_size:float=0.05):
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

def cut_geometry(geometry,block_size:float=0.05):
  block_size = 0.05
  bounds = geometry.bounds().coordinates().getInfo()[0]

  width = bounds[2][0] - bounds[0][0]
  height = bounds[2][1] - bounds[0][1]
  num_rows = math.ceil(height / block_size)
  num_cols = math.ceil(width / block_size)

  def create_blocks(row, col):
    x_min = bounds[0][0]+ col * block_size
    y_min = bounds[0][1]+ row * block_size
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

def time_difference(col, middle_date):
    '''计算middle_date与col包含日期的差值'''
    time_difference = middle_date.difference(
        ee.Date(col.get('system:time_start')), 'days').abs()
    return col.set({'time_difference': time_difference})

def rm_nodata(col, AOI):
    # 将遮罩外的元素置-99,默认的遮罩为Nodata区域，并统计Nodata的数量
    allNone_num = col.select('VV').unmask(-99).eq(-99).reduceRegion(
        **{
            'geometry': AOI,
            'reducer': ee.Reducer.sum(),
            'scale': 10,
            'maxPixels': 1e12,
            'bestEffort': True,
        }).get('VV')
    return col.set({'numNodata': allNone_num})

def Geemap_export(fileDirname,collection=False,image=False,region=None,scale=10):
    if collection:
        # 这里导出时候使用region设置AOI，否则可能因为坐标系问题(未确定)，出现黑边问题,AOI需要为长方形
        geemap.ee_export_image_collection(collection,
                        out_dir=os.path.dirname(fileDirname),
                        format = "ZIPPED_GEO_TIFF",region=region,scale=scale)
        print('collection save right')
    elif image:
        geemap.ee_export_image(image,
                    filename=fileDirname,
                    scale=scale, region=region, file_per_band=False,timeout=1500)
        print('image save right')
    else:
        print('Erro:collection && image must have one False')

def my_slope_correction(s1_ascending,s1_descending,AOI_buffer,DEM,model,Origin_scale):

    volumetric_dict = {}
    for image, orbitProperties_pass in zip([s1_ascending, s1_descending], ['ASCENDING', 'DESCENDING']):
        # orbitProperties_pass = image.get('orbitProperties_pass').getInfo()
        # get the image geometry and projection
        geom = image.geometry()
        proj = image.select(1).projection()

        azimuthEdge, rotationFromNorth, startpoint, endpoint, coordinates_dict = getASCCorners(image, AOI_buffer,
                                                                                               orbitProperties_pass)
        Heading = azimuthEdge.get('azimuth')
        Heading_Rad = ee.Image.constant(Heading).multiply(np.pi / 180)
        s1_azimuth_across = ee.Number(Heading).subtract(90.0)
        theta_iRad = image.select('angle').multiply(np.pi / 180)  # 地面入射角度转为弧度
        phi_iRad = ee.Image.constant(s1_azimuth_across).multiply(np.pi / 180)  # 方位角转弧度

        def slop_aspect(elevation, proj, geom):
            alpha_sRad = ee.Terrain.slope(elevation).select('slope').multiply(
                np.pi / 180).setDefaultProjection(proj).clip(geom)  # 坡度(与地面夹角)
            phi_sRad = ee.Terrain.aspect(elevation).select('aspect').multiply(
                np.pi / 180).setDefaultProjection(proj).clip(geom)  # 坡向角，(坡度陡峭度)坡与正北方向夹角(陡峭度)，从正北方向起算，顺时针计算角度
            phi_rRad = phi_iRad.subtract(phi_sRad)  # (飞行方向角度-坡度陡峭度)飞行方向与坡向之间的夹角
            # 分解坡度，在水平方向和垂直方向进行分解，为固定公式，cos对应水平分解，sin对应垂直分解
            alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()  # 距离向分解
            alpha_azRad = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()  # 方位向分解
            return alpha_sRad, phi_sRad, alpha_rRad, alpha_azRad

        alpha_sRad, phi_sRad, alpha_rRad, alpha_azRad = slop_aspect(DEM, proj, geom)
        sigma0Pow = ee.Image.constant(10).pow(image.divide(10.0))
        gamma0 = sigma0Pow.divide(theta_iRad.cos())  # 根据角度修订入射值
        gamma0dB = ee.Image.constant(10).multiply(gamma0.log10()).select(['VV', 'VH'],
                                                                         ['VV_gamma0', 'VH_gamma0'])  # 根据角度修订入射值
        ratio_gamma = (
            gamma0dB.select('VV_gamma0').subtract(gamma0dB.select('VH_gamma0')).rename('ratio_gamma0'))  # gamma极化相减

        def volumetric(model, theta_iRad, alpha_rRad, alpha_azRad):
            '''辐射斜率校正'''
            if model == 'volume':
                scf = volumetric_model_SCF(theta_iRad, alpha_rRad)
            if model == 'surface':
                scf = surface_model_SCF(theta_iRad, alpha_rRad, alpha_azRad)

            gamma0_flat = gamma0.divide(scf)
            gamma0_flatDB = (ee.Image.constant(10).multiply(gamma0_flat.log10()).select(['VV', 'VH'], ['VV_gamma0flat',
                                                                                                       'VH_gamma0flat']))
            ratio_flat = (gamma0_flatDB.select('VV_gamma0flat').subtract(
                gamma0_flatDB.select('VH_gamma0flat')).rename('ratio_gamma0flat'))

            return {'scf': scf, 'gamma0_flat': gamma0_flat,
                    'gamma0_flatDB': gamma0_flatDB, 'ratio_flat': ratio_flat}

        # ------------------------------RS几何畸变区域--------------------------------- 同戴可人：高山峡谷区滑坡灾害隐患InSAR早期识别
        layover = alpha_rRad.gt(theta_iRad).rename('layover')
        ninetyRad = ee.Image.constant(90).multiply(np.pi / 180)
        shadow = alpha_rRad.lt(ee.Image.constant(-1).multiply(ninetyRad.subtract(theta_iRad))).rename('shadow')

        # combine layover and shadow,因为shadow和layover都是0
        no_data_maskrgb = rgbmask(image, layover, shadow)
        slop_correction = volumetric(model, theta_iRad, alpha_rRad, alpha_azRad)

        image2 = (Eq_pixels(image.select('VV')).rename('VV_sigma0')
                  .addBands(Eq_pixels(image.select('VH')).rename('VH_sigma0'))
                  .addBands(Eq_pixels(image.select('angle')).rename('incAngle'))
                  .addBands(Eq_pixels(slop_correction['gamma0_flat'].select('VV')).rename('VV_gamma0_flat'))
                  .addBands(Eq_pixels(slop_correction['gamma0_flat'].select('VH')).rename('VH_gamma0_flat'))
                  .addBands(
            Eq_pixels(slop_correction['gamma0_flatDB'].select('VV_gamma0flat')).rename('VV_gamma0_flatDB'))
                  .addBands(
            Eq_pixels(slop_correction['gamma0_flatDB'].select('VH_gamma0flat')).rename('VH_gamma0_flatDB'))
                  .addBands(Eq_pixels(layover).rename('layover'))
                  .addBands(Eq_pixels(shadow).rename('shadow'))
                  .addBands(no_data_maskrgb)
                  .addBands(Eq_pixels(DEM.setDefaultProjection(proj).clip(geom)).rename('height')))

        cal_image = (image2.addBands(ee.Image.pixelCoordinates(proj))
                     .addBands(ee.Image.pixelLonLat()).reproject(crs=proj, scale=Origin_scale)
                     .updateMask(image2.select('VV_sigma0').mask()).clip(AOI_buffer))

        Auxiliarylines = ee.Geometry.LineString([startpoint, endpoint])

        if orbitProperties_pass == 'ASCENDING':
            volumetric_dict['ASCENDING'] = cal_image
            volumetric_dict['ASCENDING_parms'] = {'s1_azimuth_across': s1_azimuth_across,
                                                  'coordinates_dict': coordinates_dict,
                                                  'Auxiliarylines': Auxiliarylines,
                                                  'orbitProperties_pass': orbitProperties_pass,
                                                  'proj': proj}
        elif orbitProperties_pass == 'DESCENDING':
            volumetric_dict['DESCENDING'] = cal_image
            volumetric_dict['DESCENDING_parms'] = {'s1_azimuth_across': s1_azimuth_across,
                                                   'coordinates_dict': coordinates_dict,
                                                   'Auxiliarylines': Auxiliarylines,
                                                   'orbitProperties_pass': orbitProperties_pass,
                                                   'proj': proj}

    return volumetric_dict

def load_image_collection(aoi, start_date, end_date, middle_date,
                         Filter=None,FilterSize=30):
  '''s1数据加载'''
  s1_col = (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
              .filterBounds(aoi)
              .filterDate(start_date, end_date))
  s1_col_copy = copy.deepcopy(s1_col)

  # 裁剪并计算空洞数量
  # s1_col = s1_col.map(partial(clip_AOI, AOI=aoi))
  s1_col = s1_col.map(partial(rm_nodata, AOI=aoi))

  # 图像滤波，可选
  if Filter:
      print('Begin Filter ...')
      if Filter == 'leesigma':
          s1_col = s1_col.map(leesigma(FilterSize))
      elif Filter == 'RefinedLee':
          s1_col = s1_col.map(RefinedLee)
      elif Filter == 'gammamap':
          s1_col = s1_col.map(gammamap(FilterSize))
      elif Filter == 'boxcar':
          s1_col = s1_col.map(boxcar(FilterSize))
      else:
        print('Wrong Filter')
  else:
      print('Without Filter')

  # # 地形矫正，可选
  # if dem and model:
  #     print('Begin Slop Correction ...')
  #     s1_col = slope_correction(s1_col, dem, model, buffer=buffer)
  # else:
  #     # Rename = lambda image:image.rename(['VV_sigma0', 'VH_sigma0','incAngle'])
  #     # s1_col = s1_col.map(Rename)
  #     print('Without Slop Correction')

  # 计算与目标日期的差距
  s1_col = s1_col.map(partial(time_difference, middle_date=middle_date))

  # 分离升降轨
  s1_descending = s1_col.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  s1_ascending = s1_col.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))

  # 判断是否存在没有缺失值的图像,存在则筛选这些没有缺失值的图像，否则将这些图像合成为一张图像
  # 寻找与时间最接近的图像设置'synthesis': 0，否则'synthesis': 1
  filtered_collection_A = s1_ascending.filter(ee.Filter.eq('numNodata', 0))
  has_images_without_nodata_A = filtered_collection_A.size().eq(0)
  s1_ascending = ee.Algorithms.If(
      has_images_without_nodata_A,
      s1_ascending.median().reproject(s1_ascending.first().projection().crs(), None, 10).set({'synthesis': 1}),
      filtered_collection_A.filter(ee.Filter.eq('time_difference',
          s1_ascending.aggregate_min('time_difference'))).first().set({'synthesis': 0})
  )

  filtered_collection_D = s1_descending.filter(ee.Filter.eq('numNodata', 0))
  has_images_without_nodata_D = filtered_collection_D.size().eq(0)
  s1_descending = ee.Algorithms.If(
      has_images_without_nodata_D,
      s1_descending.median().reproject(s1_descending.first().projection().crs(), None, 10).set({'synthesis': 1}),
      filtered_collection_D.filter(ee.Filter.eq('time_difference',
          s1_descending.aggregate_min('time_difference'))).first().set({'synthesis': 0})
  )

  return ee.Image(s1_ascending),ee.Image(s1_descending) #,s1_col_copy

# 归一化
def afn_normalize_by_maxes(img,scale=10):
    '''
    注意：未进行重投影，max_\min_是地理坐标
    注意：使用使用
    img：要归一化的图像。
    bandMaxes：波段最大值。
    '''
    max_ = img.reduceRegion(
    reducer=ee.Reducer.max(),
    geometry=img.geometry(),
    scale=scale,
    maxPixels=1e12,
    bestEffort=True
    ).toImage().select(img.bandNames())

    min_ = img.reduceRegion(
        reducer=ee.Reducer.min(),
        geometry=img.geometry(),
        scale=scale,
        maxPixels=1e12,
        bestEffort=True
      ).toImage().select(img.bandNames()) #.reproject(img.select(0).projection().getInfo()['crs'], None, 10)

    return img.subtract(min_).divide(max_.subtract(min_))