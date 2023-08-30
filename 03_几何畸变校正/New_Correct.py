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


# 普通几何畸变校正
def cal_LIA(image, DEM, AOI_buffer, orbitProperties_pass):
    elevation = DEM
    geom = image.geometry()
    proj = image.select(1).projection()

    # Angle_aspect = ee.Terrain.aspect(image.select('angle'))
    # s1_azimuth_across = Angle_aspect.reduceRegion(ee.Reducer.mean(), geom, 1000).get('aspect')

    azimuthEdge, rotationFromNorth, startpoint, endpoint, coordinates_dict = getASCCorners(image, AOI_buffer,
                                                                                           orbitProperties_pass)
    Heading = azimuthEdge.get('azimuth')

    s1_azimuth_across = ee.Number(Heading).subtract(90.0)
    theta_iRad = image.select('angle').multiply(np.pi / 180)  # 地面入射角度转为弧度
    phi_iRad = ee.Image.constant(s1_azimuth_across).multiply(np.pi / 180)  # 方位角转弧度

    alpha_sRad = ee.Terrain.slope(elevation).select('slope').multiply(
        np.pi / 180).setDefaultProjection(proj).clip(geom)  # 坡度(与地面夹角)
    phi_sRad = ee.Terrain.aspect(elevation).select('aspect').multiply(
        np.pi / 180).setDefaultProjection(proj).clip(geom)  # 坡向角，(坡度陡峭度)坡与正北方向夹角(陡峭度)，从正北方向起算，顺时针计算角度

    height = elevation.setDefaultProjection(proj).clip(geom)

    phi_rRad = phi_iRad.subtract(phi_sRad)  # (飞行方向角度-坡度陡峭度)飞行方向与坡向之间的夹角
    # 分解坡度，在水平方向和垂直方向进行分解，为固定公式，cos对应水平分解，sin对应垂直分解
    alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()  # 距离向分解
    alpha_azRad = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()  # 方位向分解

    theta_liaRad = (alpha_azRad.cos().multiply(
        (theta_iRad.subtract(alpha_rRad)).cos())).acos()  # LIA
    # theta_liaDeg = theta_liaRad.multiply(180 / np.pi)  # LIA转弧度

    return alpha_rRad, theta_iRad, height, s1_azimuth_across,proj


def Distortion(alpha_rRad, theta_iRad, image,proj, height, AOI_buffer):
    # ------------------------------RS几何畸变区域--------------------------------- 同戴可人：高山峡谷区滑坡灾害隐患InSAR早期识别
    layover = alpha_rRad.gt(theta_iRad).rename('layover')
    ninetyRad = ee.Image.constant(90).multiply(np.pi / 180)
    shadow = alpha_rRad.lt(ee.Image.constant(-1).multiply(ninetyRad.subtract(theta_iRad))).rename('shadow')

    # # ------------------------------IJRS几何畸变区域-------------------------------
    # layover = alpha_rRad.gt(theta_iRad).rename('layover')
    # shadow = theta_liaRad.gt(ee.Image.constant(85).multiply(np.pi / 180)).rename('shadow')

    # # ------------------------------武大学报几何畸变区域---------------------------
    # layover = theta_liaRad.lt(ee.Image.constant(0).multiply(np.pi / 180)).rename('layover')
    # shadow = theta_liaRad.gt(ee.Image.constant(90).multiply(np.pi / 180)).rename('shadow')

    # RINDEX，暂时无用
    # Heading_Rad = ee.Image.constant(Heading).multiply(np.pi / 180)
    # if orbitProperties_pass == 'ASCENDING':
    #   A = phi_sRad.subtract(Heading_Rad)
    # elif orbitProperties_pass == 'DESCENDING':
    #   A = phi_sRad.add(Heading_Rad).add(np.pi)
    # R_Index = theta_iRad.subtract(alpha_sRad.multiply(A.sin()))
    # layover =

    # combine layover and shadow,因为shadow和layover都是0
    no_data_maskrgb = rgbmask(image, layover, shadow)

    image2 = (
        Eq_pixels(image.select('VV')).rename('VV_sigma0').addBands(Eq_pixels(image.select('VH')).rename('VH_sigma0'))
        .addBands(Eq_pixels(image.select('angle')).rename('incAngle')).addBands(Eq_pixels(layover).rename('layover'))
        .addBands(Eq_pixels(shadow).rename('shadow'))
        .addBands(no_data_maskrgb)
        .addBands(Eq_pixels(height).rename('height')))

    cal_image = (image2.addBands(ee.Image.pixelCoordinates(proj)).addBands(ee.Image.pixelLonLat())
                 .updateMask(image2.select('VV_sigma0').mask()).clip(AOI_buffer))
    # print("Corrected across-range-look direction", s1_azimuth_across.getInfo())
    # print("True azimuth across-range-look direction", ee.Number(Heading).subtract(90.0).getInfo())
    return cal_image



# 线性几何畸变校正
def Line_Correct(cal_image,AOI,Templist, orbitProperties_pass, proj, scale:int,cal_image_scale:int):

    line_points_list = []
    LPassive_layover_linList = []
    RPassive_layover_linList = []
    Shadow_linList = []

    for eachLine in tqdm(Templist):
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

