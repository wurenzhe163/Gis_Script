import ee
import math
import numpy as np
from scipy.signal import argrelextrema
from tqdm import tqdm
import sys


EasyIndex = lambda Data, Index, *Keys: [Data[key][Index] for key in Keys]
# 根据图像计算方位角，并
def getASCCorners(image, AOI_buffer, orbitProperties_pass):
    # 真实方位角
    coords = ee.Array(image.geometry().coordinates().get(0)).transpose()
    crdLons = ee.List(coords.toList().get(0))
    crdLats = ee.List(coords.toList().get(1))
    minLon = crdLons.sort().get(0)
    maxLon = crdLons.sort().get(-1)
    minLat = crdLats.sort().get(0)
    maxLat = crdLats.sort().get(-1)
    azimuth = (ee.Number(crdLons.get(crdLats.indexOf(minLat))).subtract(minLon).atan2(
        ee.Number(crdLats.get(crdLons.indexOf(minLon))).subtract(minLat))
               .multiply(180.0 / math.pi))

    if orbitProperties_pass == 'ASCENDING':
        azimuth = azimuth.add(270.0)
        rotationFromNorth = azimuth.subtract(360.0)
    elif orbitProperties_pass == 'DESCENDING':
        azimuth = azimuth.add(180.0)
        rotationFromNorth = azimuth.subtract(180.0)
    else:
        raise TypeError

    azimuthEdge = (ee.Feature(ee.Geometry.LineString([crdLons.get(crdLats.indexOf(minLat)), minLat, minLon,
                                                      crdLats.get(crdLons.indexOf(minLon))]),
                              {'azimuth': azimuth}).copyProperties(image))

    # 关于Buffer计算辅助线
    coords = ee.Array(image.clip(AOI_buffer).geometry().coordinates().get(0)).transpose()
    crdLons = ee.List(coords.toList().get(0))
    crdLats = ee.List(coords.toList().get(1))
    minLon = crdLons.sort().get(0)
    maxLon = crdLons.sort().get(-1)
    minLat = crdLats.sort().get(0)
    maxLat = crdLats.sort().get(-1)

    if orbitProperties_pass == 'ASCENDING':
        # 左上角
        startpoint = ee.List([minLon, maxLat])
        # 右下角
        endpoint = ee.List([maxLon, minLat])
    elif orbitProperties_pass == 'DESCENDING':
        # 右上角
        startpoint = ee.List([maxLon, maxLat])
        # 左下角
        endpoint = ee.List([minLon, minLat])

    coordinates_dict = {'crdLons': crdLons, 'crdLats': crdLats,
                        'minLon': minLon, 'maxLon': maxLon, 'minLat': minLat, 'maxLat': maxLat}

    return azimuthEdge, rotationFromNorth, startpoint, endpoint, coordinates_dict


# 将layover和shadow
def rgbmask(image, layover, shadow):
    r = ee.Image.constant(0).select([0], ['red'])
    g = ee.Image.constant(0).select([0], ['green'])
    b = ee.Image.constant(0).select([0], ['blue'])

    r = r.where(layover, 255)
    g = g.where(shadow, 255)
    return ee.Image.cat([r, g, b]).byte().updateMask(image.mask())

# 根据角度计算斜率
def angle2slope(angle):
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

def AuxiliaryLine2Point(cal_image, s1_azimuth_across,coordinates_dict, Auxiliarylines, scale):
    # 计算斜率
    K = angle2slope(s1_azimuth_across)
    # 过Auxiliarylines中的点，从最小经度到最大经度
    Max_Lon = coordinates_dict['maxLon'].getInfo()
    Min_Lon = coordinates_dict['minLon'].getInfo()

    AuxiliaryPoints = reduce_tolist(cal_image.select(['longitude', 'latitude']).clip(Auxiliarylines),
                                    scale=scale).getInfo()
    # 获取辅助线上的所有点数据
    Aux_lon = np.array(AuxiliaryPoints['longitude'])
    Aux_lon, indices_Aux_lon = np.unique(Aux_lon, return_index=True)
    Aux_lat = np.array(AuxiliaryPoints['latitude'])[indices_Aux_lon]

    Templist = []
    for X, Y in zip(Aux_lon, Aux_lat):
        C = Y - K * X
        Min_Lon_Y = K * Min_Lon + C
        Max_lon_Y = K * Max_Lon + C
        Templist.append(ee.Feature(ee.Geometry.LineString(
            [Min_Lon, Min_Lon_Y, Max_Lon, Max_lon_Y])))
    return Templist

# 将图像像素数量与经纬度等匹配
def Eq_pixels(x): return ee.Image.constant(0).where(x, x).updateMask(x.mask())

# 将图像转为点数据表达
def reduce_tolist(each, scale): return ee.Image(each).reduceRegion(
    reducer=ee.Reducer.toList(), geometry=each.geometry(), scale=scale, maxPixels=1e13)

# 线性几何畸变校正
def Line_Correct(cal_image,AOI,Templist, orbitProperties_pass, proj, scale:int,cal_image_scale:int):

    line_points_list = []
    LPassive_layover_linList = []
    RPassive_layover_linList = []
    Shadow_linList = []

    for eachLine in Templist:
        # 求与辅助线相交的像素值,求取极值点位置
        LineImg = cal_image.select(
            ['height', 'layover', 'shadow', 'incAngle', 'x', 'y', 'longitude', 'latitude']).clip(eachLine)
        LineImg_point = LineImg.reduceRegion(
            reducer=ee.Reducer.toList(),
            geometry=cal_image.geometry(),
            scale=scale,
            maxPixels=1e13)
        line_points_list.append(LineImg_point)
    list_of_dicts = ee.List(line_points_list).getInfo()  # 需转换数据到本地，严重耗时

    for PointDict in tqdm(list_of_dicts):
        if orbitProperties_pass == 'ASCENDING':
            order = np.argsort(PointDict['x'])
        elif orbitProperties_pass == 'DESCENDING':
            order = np.argsort(PointDict['x'])[::-1]
        PointDict = {k: np.array(v)[order] for k, v in PointDict.items()}
        PointDict['x'], PointDict['y'] = PointDict['x'] * \
                                  cal_image_scale, PointDict['y'] * cal_image_scale  # 像素行列10m分辨率，由proj得

        # 寻找入射线上DEM的极大值点，Asending左侧可能出现layover，右侧可能出现shadow。  Decending反之

        index_max = argrelextrema(PointDict['height'], np.greater)[0]
        Angle_max, Z_max, X_max, Y_max = EasyIndex(
            PointDict, index_max, 'incAngle', 'height', 'x', 'y')

        # --------检索出一条线上所有Peak点的畸变
        LPassive_layover = []
        RPassive_layover = []
        Passive_shadow = []
        for each in range(len(index_max)):
            '''
            rx->r_angle , Peak点对应的数值
            Lay|L , 左侧叠掩
            Rlay ， 右侧叠掩
            '''
            rx = X_max[each]
            ry = Y_max[each]
            rh = Z_max[each]
            rindex = index_max[each]
            r_angle = Angle_max[each]

            # 被动叠掩（仅山底）
            if index_max[each] - 50 > 0:
                rangeIndex = range(index_max[each] - 50, index_max[each])
            else:
                rangeIndex = range(0, index_max[each])

            L_h, L_x, L_y, L_lon, L_lat, L_angle = EasyIndex(
                PointDict, rangeIndex, 'height', 'x', 'y', 'longitude', 'latitude', 'incAngle')

            Llay_angle_iRad = np.arctan(
                (rh - L_h) / np.sqrt(np.square(L_x - rx) + np.square(L_y - ry)))
            Llay_angle = Llay_angle_iRad * 180 / math.pi
            index_Llay = np.where(Llay_angle > r_angle)[0]

            if len(index_Llay) != 0:
                tlon_Llay = L_lon[index_Llay]
                tlat_Llay = L_lat[index_Llay]
                LlayFeatureCollection = ee.FeatureCollection([ee.Feature(
                    ee.Geometry.Point(x, y), {'values': 1}) for x, y in zip(tlon_Llay, tlat_Llay)])
                # 将属性值映射到图像上并设置默认值
                LPassive_layover.append(
                    LlayFeatureCollection.reduceToImage(['values'], 'mean'))
                # image_with_values = image_with_values.paint(TempFeatureCollection,'values')

            # 阴影
            if index_max[each] + 50 < len(PointDict['x']):
                rangeIndex = range(index_max[each]+1, index_max[each] + 50)
            else:
                rangeIndex = range(index_max[each]+1, len(PointDict['x']))

            R_h, R_x, R_y, R_lon, R_lat = EasyIndex(
                PointDict, rangeIndex, 'height', 'x', 'y', 'longitude', 'latitude')
            R_angle_iRad = np.arctan(
                (rh - R_h) / np.sqrt(np.square(R_x - rx) + np.square(R_y - ry)) + sys.float_info.min)
            R_angle = R_angle_iRad * 180 / math.pi
            index_Shadow = np.where(R_angle > (90 - r_angle))[0]

            if len(index_Shadow) != 0:
                # 阴影
                tlon_Shadow = R_lon[index_Shadow]
                tlat_Shadow = R_lat[index_Shadow]
                ShadowFeatureCollection = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(
                    x, y), {'values': 1}) for x, y in zip(tlon_Shadow, tlat_Shadow)])
                Passive_shadow.append(
                    ShadowFeatureCollection.reduceToImage(['values'], 'mean'))

            if len(index_Llay) != 0:
                # 被动叠掩(山顶右侧),与阴影运算交叉
                layoverM_x, layoverM_y, layoverM_h, layoverM_angle = \
                    L_x[index_Llay[-1]], L_y[index_Llay[-1]], L_h[index_Llay[-1]], L_angle[index_Llay[-1]]  # 起算点
                Rlay_angle_iRad = np.arctan(
                    (R_h - layoverM_h) / np.sqrt(np.square(R_x - layoverM_x) + np.square(R_y - layoverM_y)))
                Rlay_angle = Rlay_angle_iRad * 180 / math.pi
                index_Rlayover = np.where(Rlay_angle > layoverM_angle)[0]
                if len(index_Rlayover) != 0:
                    tlon_RLay = R_lon[index_Rlayover]
                    tlat_RLay = R_lat[index_Rlayover]
                    RLayFeatureCollection = ee.FeatureCollection([ee.Feature(
                        ee.Geometry.Point(x, y), {'values': 1}) for x, y in zip(tlon_RLay, tlat_RLay)])
                    RPassive_layover.append(
                        RLayFeatureCollection.reduceToImage(['values'], 'mean'))

        if len(LPassive_layover) != 0:
            aggregated_image = ee.ImageCollection(
                LPassive_layover).mosaic().reproject(crs=proj, scale=scale)
            LPassive_layover_linList.append(aggregated_image)

        if len(RPassive_layover) != 0:
            aggregated_image = ee.ImageCollection(
                RPassive_layover).mosaic().reproject(crs=proj, scale=scale)
            RPassive_layover_linList.append(aggregated_image)

        if len(Passive_shadow) != 0:
            aggregated_image = ee.ImageCollection(
                Passive_shadow).mosaic().reproject(crs=proj, scale=scale)
            Shadow_linList.append(aggregated_image)

    LeftLayover = ee.ImageCollection(LPassive_layover_linList).mosaic().reproject(crs=proj, scale=scale).clip(AOI)
    RightLayover = ee.ImageCollection(RPassive_layover_linList).mosaic().reproject(crs=proj, scale=scale).clip(AOI)
    Shadow = ee.ImageCollection(Shadow_linList).mosaic().reproject(crs=proj, scale=scale).clip(AOI)

    return LeftLayover.toInt8(), RightLayover.toInt8(), Shadow.toInt8()

