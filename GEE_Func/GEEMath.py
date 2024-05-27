import ee,math
import numpy as np
from PackageDeepLearn.utils.Statistical_Methods import Cal_HistBoundary
def calculate_iou(geometry1, geometry2):
    intersection = geometry1.intersection(geometry2)
    union = geometry1.union(geometry2)
    intersection_area = intersection.area()
    union_area = union.area()
    return intersection_area.divide(union_area)

def get_minmax(Image: ee.Image,AOI=False, scale: int = 10):
    '''Image 只能包含有一个波段，否则不能重命名'''
    if AOI:
        geometry = AOI
    else:
        geometry = Image.geometry()
    Obj = Image.reduceRegion(reducer=ee.Reducer.minMax(), geometry=geometry, scale=scale, bestEffort=True)
    return Obj.rename(**{'from': Obj.keys(), 'to': ['max', 'min']})

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

# 归一化
def normalize_by_minmax(img, scale=10):
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
    ).toImage().select(img.bandNames())  # .reproject(img.select(0).projection().getInfo()['crs'], None, 10)

    return img.subtract(min_).divide(max_.subtract(min_))

def get_meanStd(Image: ee.Image, scale: int = 10):
    '''Image 只能包含有一个波段，否则不能重命名'''
    Obj = Image.reduceRegion(reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                             geometry=Image.geometry(), scale=scale, bestEffort=True)
    return Obj.rename(**{'from': Obj.keys(), 'to': ['mean', 'std']})

def get_histogram(Image: ee.Image, region, scale, histNum=1000):
    histogram0 = Image.reduceRegion(
        reducer=ee.Reducer.histogram(histNum),
        geometry=region,
        scale=scale,
        maxPixels=1e12,
        bestEffort=True)  # 自动平衡scale减少计算量
    histogram = histogram0.get(Image.bandNames().get(0))
    return histogram

def get_histAndboundary(Image: ee.Image, region, scale, histNum=1000, y=100):
    '''适用于GEE'''
    histogram = get_histogram(Image, region, scale, histNum=histNum).getInfo()
    bin_centers, counts = (np.array(histogram['bucketMeans']), np.array(histogram['histogram']))
    HistBound = Cal_HistBoundary(counts, y=y)
    bin_centers = bin_centers[HistBound['indexFront']:HistBound['indexBack']]
    counts = counts[HistBound['indexFront']:HistBound['indexBack']]
    return bin_centers, counts, histogram['bucketWidth']

def histogramMatching(sourceImg, targetImg, AOI, source_bandsNames, target_bandsNames, Histscale=30, maxBuckets=256):
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
    assert len(source_bandsNames) == len(
        target_bandsNames), 'source and target image must have the same number of bands'

    args = {
        'reducer': ee.Reducer.autoHistogram(**{'maxBuckets': maxBuckets, 'cumulative': True}),
        'geometry': AOI,
        'scale': Histscale,
        'maxPixels': 1e13,
        'tileScale': 16
    }
    source = sourceImg.reduceRegion(**args)
    target = targetImg.updateMask(sourceImg.mask()).reduceRegion(**args)

    Copy_sourceImg = []
    for band_source, band_target in zip(source_bandsNames, target_bandsNames):
        Lookup = lookup(source.getArray(band_source), target.getArray(band_target))
        Copy_sourceImg.append(sourceImg.select([band_source]).interpolate(**Lookup))
    return ee.Image.cat(Copy_sourceImg)

def time_difference(col, middle_date,timeCol='system:time_start',time='days'):
    '''计算middle_date与col包含日期的差值'''
    time_difference = middle_date.difference(
        ee.Date(col.get(timeCol)), time).abs()
    return col.set({timeCol: time_difference})

# 根据角度计算斜率
def angle2slope(angle):
    # 判断角度所在的范围并计算斜率
    def compute_slope(ang):
        # 将角度调整为特定范围内的有效值
        adjusted_angle = ee.Number(ee.Algorithms.If(ang.gt(180), ee.Number(90).subtract(ang.subtract(180)), ang))
        adjusted_angle = ee.Number(ee.Algorithms.If(ang.gt(90).And(ang.lte(180)), ang.subtract(90), adjusted_angle))
        adjusted_angle = ee.Number(ee.Algorithms.If(ang.gt(270).And(ang.lte(360)), ang.subtract(270), adjusted_angle))
        
        # 转换角度为弧度
        radians = adjusted_angle.multiply(ee.Number(math.pi / 180))
        
        # 计算斜率
        slope = radians.tan()
        
        # 根据角度范围调整斜率的符号
        slope = ee.Number(ee.Algorithms.If(ang.gt(90).And(ang.lte(180)), slope.multiply(-1), slope))
        slope = ee.Number(ee.Algorithms.If(ang.gt(270).And(ang.lte(360)), slope.multiply(-1), slope))
        
        return slope

    # 检查angle是否为ee.Number类型
    if isinstance(angle, ee.ee_number.Number):
        # 直接计算斜率
        return compute_slope(angle)
    else:
        # 如果不是ee.Number，假设它是可直接处理的数值
        angle = ee.Number(angle)
        return compute_slope(angle)
    
def angle2slope_numpy(angle):
    if type(angle) == ee.ee_number.Number:
        angle = angle.getInfo()
    if 0 < angle <= 90 or 180 < angle <= 270:
        if 180 < angle <= 270:
            angle = 90 - (angle - 180)
        arc = angle / 180 * math.pi
        slop = math.tan(arc)
    elif 90 < angle <= 180 or 270 < angle <= 360:
        if 90 < angle <= 180:
            angle = angle - 90
        elif 270 < angle <= 360:
            angle = angle - 270
        arc = angle / 180 * math.pi
        slop = -math.tan(arc)
    return slop