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
def Line_Correct2(cal_image,AOI,Templist, orbitProperties_pass, proj, scale:int,cal_image_scale:int):

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

# 被动几何畸变识别（方法2）
def Line_Correct_test(cal_image,AOI_buffer,Templist, orbitProperties_pass, proj, scale:int):
    line_points_list = []
    # LPassive_layover_linList = []
    # RPassive_layover_linList = []
    # Shadow_linList = []

    for eachLine in tqdm(Templist):
        # 求与辅助线相交的像素值,求取极值点位置
        LineImg = cal_image.select(
            ['height', 'layover', 'shadow', 'incAngle', 'x', 'y', 'longitude', 'latitude']).clip(eachLine)
        LineImg_point = LineImg.reduceRegion(
            reducer=ee.Reducer.toList(),
            geometry=cal_image.geometry(),
            scale=30,
            maxPixels=1e13)
        line_points_list.append(LineImg_point)
    list_of_dicts = ee.List(line_points_list).getInfo()  # 需转换数据到本地，严重耗时
    # 分别为右侧被动叠掩点、左侧被动叠掩点和被动阴影点经纬度列表
    r_lon_sum = []
    r_lat_sum = []
    l_lon_sum = []
    l_lat_sum = []
    sh_lon_sum = []
    sh_lat_sum = []
    # 逐视线检索被动畸变
    for PointDict in tqdm(list_of_dicts):
        if orbitProperties_pass == 'ASCENDING':
            order = np.argsort(PointDict['x'])
        elif orbitProperties_pass == 'DESCENDING':
            order = np.argsort(PointDict['x'])[::-1]
        PointDict = {k: np.array(v)[order] for k, v in PointDict.items()}  # 将视线上的像元按照x，从小到大排序
        PointDict['x'], PointDict['y'] = PointDict['x'] * \
                                         10, PointDict['y'] * 10  # 像素行列10m分辨率，由proj得
        EasyIndex = lambda Data, Index, *Keys: [Data[key][Index] for key in Keys]
        # 被动阴影像元识别
        if len(np.where(PointDict['shadow'] == 1)[0]) != 0:
            # 利用grad_fn函数识别出主动阴影起始索引和结束索引
            shadow_grad_fn = lambda shadow_array: [shadow_array[i + 1] - shadow_array[i] for i in
                                                   range(len(shadow_array) - 1)]
            PointDict_shadow_padding = np.insert(PointDict['shadow'], 0, 0)
            shadow_grad = np.array(shadow_grad_fn(PointDict_shadow_padding))
            # 防止PointDict['shadow']最后是以1结尾，导致-1比1个数少一个，且1结尾没有检测被动阴影必要，所以直接去掉
            if PointDict_shadow_padding[-1] == 1:
                shadow_start_index = np.where(shadow_grad == 1)[0][:-1]
            else:
                shadow_start_index = np.where(shadow_grad == 1)[0]
            shadow_range_index = np.where(shadow_grad == -1)[0]
            # 以主动阴影的起始像元位置为参考点
            sh_h, sh_x, s_y, sh_lon, sh_lat, sh_angle = EasyIndex(
                PointDict, shadow_start_index, 'height', 'x', 'y', 'longitude', 'latitude', 'incAngle')
            # 在某一视线中，逐段主动阴影进行分析
            if len(shadow_start_index) != 0:
                for i in range(shadow_start_index.size):
                    start_sh_index = shadow_start_index[i]
                    range_sh_index = shadow_range_index[i]
                    index_Shadow = []
                    # 确定检索像元范围
                    if range_sh_index + 50 < len(PointDict['shadow']):
                        rangeIndex_shadow = range(range_sh_index, range_sh_index + 50)
                    else:
                        rangeIndex_shadow = range(range_sh_index, len(PointDict['shadow']))
                    if len(rangeIndex_shadow) != 0:
                        sh_Range_h, sh_Range_x, sh_Range_y, sh_Range_lon, sh_Range_lat = EasyIndex(
                            PointDict, rangeIndex_shadow, 'height', 'x', 'y', 'longitude', 'latitude')
                        shadow_angle_iRad = np.arctan((sh_h[i] - sh_Range_h) / np.sqrt(
                            np.square(sh_Range_x - sh_x[i]) + np.square(sh_Range_y - s_y[i])))
                        shadow_angle = shadow_angle_iRad * 180 / math.pi
                        index_Shadow = np.where(shadow_angle > (90 - sh_angle[i]))[0]

                    if len(index_Shadow) != 0:
                        tlon_shadow = sh_Range_lon[index_Shadow]
                        tlat_shadow = sh_Range_lat[index_Shadow]
                        for j in range(len(tlon_shadow)):
                            sh_lon_sum.append(tlon_shadow[j])
                            sh_lat_sum.append(tlat_shadow[j])

        # 左右侧被动叠掩
        if len(np.where(PointDict['layover'] == 1)[0]) != 0:
            layover_grad_fn = lambda lay_array: [lay_array[i + 1] - lay_array[i] for i in range(len(lay_array) - 1)]
            PointDict_lay_padding = np.insert(PointDict['layover'], 0, 0)
            # PointDict_lay_padding=np.append(PointDict_lay_padding,0)
            layover_grad = np.array(layover_grad_fn(PointDict_lay_padding))
            if PointDict_lay_padding[-1] == 1:
                rlayover_start_index = np.where(layover_grad == 1)[0][:-1]
            else:
                rlayover_start_index = np.where(layover_grad == 1)[0]
            rlayover_range_index = np.where(layover_grad == -1)[0]
            # 右侧叠掩以rlayover_start_index为参考点，rlayover_range_index为像元检索起点，左侧叠掩相反
            s_h, s_x, s_y, s_lon, s_lat, s_angle = EasyIndex(
                PointDict, rlayover_start_index, 'height', 'x', 'y', 'longitude', 'latitude', 'incAngle')
            d_h, d_x, d_y, d_lon, d_lat, d_angle = EasyIndex(
                PointDict, rlayover_range_index, 'height', 'x', 'y', 'longitude', 'latitude', 'incAngle')
            # 同时进行左右俩侧像元叠掩检测
            for i in range(rlayover_start_index.size):
                start_index = rlayover_start_index[i]
                range_index = rlayover_range_index[i]
                index_Rlayover = []
                index_Llayover = []
                if range_index + 50 < len(PointDict['layover']):
                    rangeIndex = range(range_index, range_index + 50)
                else:
                    rangeIndex = range(range_index, len(PointDict['layover']))

                if start_index - 50 > 0:
                    rangeIndex_l = range(start_index - 50, start_index)
                else:
                    rangeIndex_l = range(start_index)
                # 为检测出完整的被动叠掩位置，像元应同时满足其与主动叠掩起始位置和终止位置的几何关系
                if len(rangeIndex) != 0:
                    r_Range_h, r_Range_x, r_Range_y, r_Range_lon, r_Range_lat = EasyIndex(
                        PointDict, rangeIndex, 'height', 'x', 'y', 'longitude', 'latitude')
                    Rlay_start_iRad = np.arctan(
                        (r_Range_h - s_h[i]) / np.sqrt(np.square(r_Range_x - s_x[i]) + np.square(r_Range_y - s_y[i])))
                    Rlay_end_iRad = np.arctan(
                        (r_Range_h - d_h[i]) / np.sqrt(np.square(r_Range_x - d_x[i]) + np.square(r_Range_y - d_y[i])))
                    Rlay_start = Rlay_start_iRad * 180 / math.pi
                    Rlay_end = Rlay_end_iRad * 180 / math.pi
                    index_Rlayover = np.where(np.logical_and(Rlay_start > s_angle[i], Rlay_end < d_angle[i]))[0]

                if len(rangeIndex_l) != 0:
                    l_Range_h, l_Range_x, l_Range_y, l_Range_lon, l_Range_lat = EasyIndex(
                        PointDict, rangeIndex_l, 'height', 'x', 'y', 'longitude', 'latitude')
                    Llay_start_iRad = np.arctan(
                        (s_h[i] - l_Range_h) / np.sqrt(np.square(s_x[i] - l_Range_x) + np.square(s_y[i] - l_Range_y)))
                    Llay_end_iRad = np.arctan(
                        (d_h[i] - l_Range_h) / np.sqrt(np.square(d_x[i] - l_Range_x) + np.square(d_y[i] - l_Range_y)))
                    Llay_start = Llay_start_iRad * 180 / math.pi
                    Llay_end = Llay_end_iRad * 180 / math.pi
                    index_Llayover = np.where(np.logical_and(Llay_start < s_angle[i], Llay_end > d_angle[i]))[0]

                if len(index_Rlayover) != 0:
                    tlon_RLay = r_Range_lon[index_Rlayover]
                    tlat_RLay = r_Range_lat[index_Rlayover]
                    for j in range(len(tlat_RLay)):
                        r_lon_sum.append(tlon_RLay[j])
                        r_lat_sum.append(tlat_RLay[j])
                if len(index_Llayover) != 0:
                    tlon_LLay = l_Range_lon[index_Llayover]
                    tlat_LLay = l_Range_lat[index_Llayover]
                    for j in range(len(tlat_LLay)):
                        l_lon_sum.append(tlon_LLay[j])
                        l_lat_sum.append(tlat_LLay[j])

    rlay_featurecollection = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point(x, y), {'values': 5}) for x, y in zip(r_lon_sum, r_lat_sum)])
    shadow_featurecollection = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point(x, y), {'values': 3}) for x, y in zip(sh_lon_sum, sh_lat_sum)])
    llay_featurecollection = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point(x, y), {'values': 4}) for x, y in zip(l_lon_sum, l_lat_sum)])
    image_rlayover = ee.Image().paint(rlay_featurecollection, 'values')
    image_llayover = ee.Image().paint(llay_featurecollection, 'values')
    image_shadow = ee.Image().paint(shadow_featurecollection, 'values')
    img_rlayover = image_rlayover.clip(AOI_buffer).reproject(crs=proj, scale=scale)
    img_llayover = image_llayover.clip(AOI_buffer).reproject(crs=proj, scale=scale)
    img_shadow = image_shadow.clip(AOI_buffer).reproject(crs=proj, scale=scale)
    passive_img = ee.ImageCollection([img_rlayover, img_llayover, img_shadow]).mosaic()
    return passive_img