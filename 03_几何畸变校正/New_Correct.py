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
def rgbmask(image, **parms):
    r = ee.Image.constant(0).select([0], ['red'])
    g = ee.Image.constant(0).select([0], ['green'])
    b = ee.Image.constant(0).select([0], ['blue'])

    for key, value in parms.items():
        lenName = value.bandNames().length().getInfo()
        if lenName:
            # print('lenName={}'.format(lenName))
            if key == 'layover' or key == 'Llayover':
                # print('r,key={}'.format(key))
                r = r.where(value, 255)
            if key == 'shadow':
                # print('g,key={}'.format(key))
                g = g.where(value, 255)
            if key == 'Rlayover':
                # print('b,key={}'.format(key))
                b = b.where(value, 255)
        else:
            continue
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


def AuxiliaryLine2Point(cal_image, s1_azimuth_across, coordinates_dict, Auxiliarylines, scale):
    '''获取所有待计算的计算线'''
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
def Line_Correct_old(cal_image, AOI, Templist, orbitProperties_pass, proj, scale: int, cal_image_scale: int,
                 filt_distance=False, save_peak=False, line_points_connect=False, Peak_Llay=False, Peak_shdow=False,
                 Peak_Rlay=False):
    '''
    filt_distance: int, 米。是否过滤距离小于filt_distance的点（减缓速度）
    save_peak: bool, 是否保存Peak点（减缓速度）
    line_points_connect: bool, 是否连线（减缓速度）
    Peak_Llay: bool, Llay是否包含peak点
    Peak_shdow: bool, 阴影是否包含peak点
    Peak_Rlay: bool, Rlay是否包含peak点
    '''

    line_points_list = []
    LPassive_layover_linList = []
    RPassive_layover_linList = []
    Shadow_linList = []
    # PeakPoint_list = []

    for eachLine in tqdm(Templist):
        # 求与辅助线相交的像素值
        LineImg = cal_image.select(
            ['height', 'layover', 'shadow', 'incAngle', 'alpha_rRad', 'x', 'y', 'longitude', 'latitude']).clip(eachLine)
        # 转为Dictionary
        ptsDict = LineImg.reduceRegion(
            reducer=ee.Reducer.toList(),
            geometry=cal_image.geometry(),
            scale=scale,
            maxPixels=1e13)
        if filt_distance:
            # 抽取经纬度，转点
            lons = ee.List(ptsDict.get('longitude'))
            lats = ee.List(ptsDict.get('latitude'))
            Point_list = ee.FeatureCollection(lons.zip(lats).map(lambda xy: ee.Feature(ee.Geometry.Point(xy))))

            # 计算到eachLine的距离
            Point_list = Point_list.map(lambda f: f.set('dis', eachLine.geometry().distance(f.geometry())))
            distances = ee.List(Point_list.reduceColumns(ee.Reducer.toList(), ['dis']).get('list'))

            # 过滤 Distance 列表小于 10 的元素
            filteredDistances = ee.List(distances.filter(ee.Filter.lt('item', filt_distance)))

            # 计算索引，选择索引
            Index_filter = filteredDistances.map(lambda x: distances.indexOf(x))
            ptsDict = ptsDict.map(lambda k, v: Index_filter.map(lambda x: ee.List(v).get(x))).set('Distance',
                                                                                                  filteredDistances)
        line_points_list.append(ptsDict)

    list_of_dicts = ee.List(line_points_list).getInfo()  # 需转换数据到本地，严重耗时
    # print('line_points_list元素数量={}'.format(sum([len(list_of_dicts[i]['longitude']) for i in range(len(list_of_dicts))])))

    for PointDict in tqdm(list_of_dicts):
        if orbitProperties_pass == 'ASCENDING':
            order = np.argsort(PointDict['longitude'])
        elif orbitProperties_pass == 'DESCENDING':
            order = np.argsort(PointDict['longitude'])[::-1]

        PointDict = {k: np.array(v)[order] for k, v in PointDict.items()}
        PointDict['x'], PointDict['y'] = PointDict['x'] * cal_image_scale, PointDict[
            'y'] * cal_image_scale  # 像素行列10m分辨率，由proj得

        # 寻找入射线上DEM的极大值点
        index_max = argrelextrema(PointDict['height'], np.greater)[0]
        if len(index_max) != 0:
            Angle_max, Z_max, X_max, Y_max, Lon_max, Lat_max = EasyIndex(
                PointDict, index_max, 'incAngle', 'height', 'x', 'y', 'longitude', 'latitude')

            if save_peak:
                # 保存 DEM极值点
                # for x, y in zip(Lon_max, Lat_max):
                #     PeakPoint_list.append(ee.Feature(ee.Geometry.Point(x, y), {'values': 1}))
                if 'PeakFeatureCollection' in locals():
                    PeakFeatureCollection = PeakFeatureCollection.merge(ee.FeatureCollection([ee.Feature(
                        ee.Geometry.Point(x, y), {'values': 1}) for x, y in zip(Lon_max, Lat_max)]))
                else:
                    PeakFeatureCollection = ee.FeatureCollection([ee.Feature(
                        ee.Geometry.Point(x, y), {'values': 1}) for x, y in zip(Lon_max, Lat_max)])

            # --------检索出一条线上所有Peak点的畸变
            LPassive_layover = []
            RPassive_layover = []
            Passive_shadow = []

            for each in range(len(index_max)):

                rx = X_max[each]
                ry = Y_max[each]
                rh = Z_max[each]
                rlon = Lon_max[each]
                rlat = Lat_max[each]
                rindex = index_max[each]
                r_angle = Angle_max[each]

                Pixels_cal = 900 // scale

                # 被动叠掩（仅山底）
                if index_max[each] - Pixels_cal > 0:
                    rangeIndex = range(rindex - Pixels_cal, rindex)
                else:
                    rangeIndex = range(0, rindex)

                ## 计算坡度梯度会少一位，因此在第一位进行补‍0，保持元素数量一致。计算坡度梯度变化，变化最大点，即为连接地面点
                PointDict['Grad_alpha_rRad'] = np.insert(np.diff(PointDict['alpha_rRad']), 0, 0)
                L_h, L_x, L_y, L_lon, L_lat, L_angle, L_Grad_alpha_rRad = EasyIndex(
                    PointDict, rangeIndex, 'height', 'x', 'y', 'longitude', 'latitude', 'incAngle', 'Grad_alpha_rRad')

                Llay_angle_iRad = np.arctan((rh - L_h) / np.sqrt(np.square(L_x - rx) + np.square(L_y - ry)))
                Llay_angle = Llay_angle_iRad * 180 / math.pi

                index_Llay = np.where(Llay_angle > r_angle)[0]

                if len(index_Llay) != 0:
                    if line_points_connect:
                        index_Llay = range(np.where(Llay_angle > r_angle)[0][0], len(Llay_angle))
                    tlon_Llay = L_lon[index_Llay]
                    tlat_Llay = L_lat[index_Llay]
                    if Peak_Llay:
                        tlon_Llay = np.append(tlon_Llay, rlon)
                        tlat_Llay = np.append(tlat_Llay, rlat)

                    LlayFeatureCollection = ee.FeatureCollection([ee.Feature(
                        ee.Geometry.Point(x, y), {'values': 1}) for x, y in zip(tlon_Llay, tlat_Llay)])
                    # 将属性值映射到图像上并设置默认值
                    LPassive_layover.append(
                        LlayFeatureCollection.reduceToImage(['values'], 'mean'))
                    # image_with_values = image_with_values.paint(TempFeatureCollection,'values')

                # 阴影
                if index_max[each] + Pixels_cal < len(PointDict['x']):
                    rangeIndex = range(index_max[each] + 1, index_max[each] + Pixels_cal)
                else:
                    rangeIndex = range(index_max[each] + 1, len(PointDict['x']))

                R_h, R_x, R_y, R_lon, R_lat = EasyIndex(
                    PointDict, rangeIndex, 'height', 'x', 'y', 'longitude', 'latitude')
                R_angle_iRad = np.arctan(
                    (rh - R_h) / np.sqrt(np.square(R_x - rx) + np.square(R_y - ry)) + sys.float_info.min)
                R_angle = R_angle_iRad * 180 / math.pi
                index_Shadow = np.where(R_angle > (90 - r_angle))[0]

                if len(index_Shadow) != 0:
                    if line_points_connect:
                        index_Shadow = range(0, np.where(R_angle > (90 - r_angle))[0][-1])
                    tlon_Shadow = R_lon[index_Shadow]
                    tlat_Shadow = R_lat[index_Shadow]
                    if Peak_shdow:
                        tlon_Shadow = np.append(tlon_Shadow, rlon)
                        tlat_Shadow = np.append(tlat_Shadow, rlat)
                    ShadowFeatureCollection = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(
                        x, y), {'values': 1}) for x, y in zip(tlon_Shadow, tlat_Shadow)])
                    Passive_shadow.append(
                        ShadowFeatureCollection.reduceToImage(['values'], 'mean'))

                if len(index_Llay) != 0:
                    # 被动叠掩(山顶右侧),与阴影运算交叉
                    # 求取在左侧叠掩范围内变化最大值点,在index_Llay范围内求最大坡度变化值
                    try:
                        L_bottom = index_Llay[np.argmax(L_Grad_alpha_rRad[index_Llay])]
                        # layoverM_x, layoverM_y = L_x[L_bottom], L_y[L_bottom],  # ,L_lon[L_bottom], L_lat[L_bottom]
                        layoverM_x, layoverM_y = rx, ry
                        layoverM_h = L_h[L_bottom]  # np.min(L_h[index_Llay])
                        layoverM_angle = r_angle
                    except:
                        raise IndexError('索引问题each={}'.format(each))

                    Rlay_angle_iRad = np.arctan(
                        (R_h - layoverM_h) / np.sqrt(np.square(R_x - layoverM_x) + np.square(R_y - layoverM_y)))
                    Rlay_angle = Rlay_angle_iRad * 180 / math.pi
                    index_Rlayover = np.where(Rlay_angle > layoverM_angle)[0]

                    if len(index_Rlayover) != 0:
                        if line_points_connect:
                            index_Rlayover = range(0, np.where(Rlay_angle > layoverM_angle)[0][-1])

                        tlon_RLay = R_lon[index_Rlayover]
                        tlat_RLay = R_lat[index_Rlayover]
                        if Peak_Rlay:
                            tlon_RLay = np.append(tlon_RLay, rlon)
                            tlat_RLay = np.append(tlat_RLay, rlat)

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
    if save_peak:
        return LeftLayover.toInt8(), RightLayover.toInt8(), Shadow.toInt8(), PeakFeatureCollection
    else:
        return LeftLayover.toInt8(), RightLayover.toInt8(), Shadow.toInt8()

# 线性几何畸变校正
def Line_Correct(cal_image, AOI_buffer, Templist, orbitProperties_pass, proj, scale: int, cal_image_scale: int,
                 filt_distance=False, save_peak=False, line_points_connect=False, Peak_Llay=False, Peak_shdow=False,
                 Peak_Rlay=False):
    '''
    filt_distance: int, 米。是否过滤距离小于filt_distance的点（减缓速度）
    save_peak: bool, 是否保存Peak点（减缓速度）
    line_points_connect: bool, 是否连线（减缓速度）
    Peak_Llay: bool, Llay是否包含peak点
    Peak_shdow: bool, 阴影是否包含peak点
    Peak_Rlay: bool, Rlay是否包含peak点
    '''

    line_points_list = []
    LPassive_layover = []
    RPassive_layover = []
    Passive_shadow = []

    for eachLine in tqdm(Templist):
        # 求与辅助线相交的像素值
        LineImg = cal_image.select(
            ['height', 'layover', 'shadow', 'incAngle', 'alpha_rRad', 'x', 'y', 'longitude', 'latitude']).clip(eachLine)
        # 转为Dictionary
        ptsDict = LineImg.reduceRegion(
            reducer=ee.Reducer.toList(),
            geometry=cal_image.geometry(),
            scale=scale,
            maxPixels=1e13)
        if filt_distance:
            # 抽取经纬度，转点
            lons = ee.List(ptsDict.get('longitude'))
            lats = ee.List(ptsDict.get('latitude'))
            Point_list = ee.FeatureCollection(lons.zip(lats).map(lambda xy: ee.Feature(ee.Geometry.Point(xy))))

            # 计算到eachLine的距离
            Point_list = Point_list.map(lambda f: f.set('dis', eachLine.geometry().distance(f.geometry())))
            distances = ee.List(Point_list.reduceColumns(ee.Reducer.toList(), ['dis']).get('list'))

            # 过滤 Distance 列表小于 10 的元素
            filteredDistances = ee.List(distances.filter(ee.Filter.lt('item', filt_distance)))

            # 计算索引，选择索引
            Index_filter = filteredDistances.map(lambda x: distances.indexOf(x))
            ptsDict = ptsDict.map(lambda k, v: Index_filter.map(lambda x: ee.List(v).get(x))).set('Distance',
                                                                                                  filteredDistances)
        line_points_list.append(ptsDict)

    list_of_dicts = ee.List(line_points_list).getInfo()  # 需转换数据到本地，严重耗时
    # print('line_points_list元素数量={}'.format(sum([len(list_of_dicts[i]['longitude']) for i in range(len(list_of_dicts))])))

    for PointDict in tqdm(list_of_dicts):
        if orbitProperties_pass == 'ASCENDING':
            order = np.argsort(PointDict['longitude'])
        elif orbitProperties_pass == 'DESCENDING':
            order = np.argsort(PointDict['longitude'])[::-1]

        PointDict = {k: np.array(v)[order] for k, v in PointDict.items()}
        PointDict['x'], PointDict['y'] = PointDict['x'] * cal_image_scale, PointDict[
            'y'] * cal_image_scale  # 像素行列10m分辨率，由proj得

        # 寻找入射线上DEM的极大值点
        index_max = argrelextrema(PointDict['height'], np.greater)[0]
        if len(index_max) != 0:
            Angle_max, Z_max, X_max, Y_max, Lon_max, Lat_max = EasyIndex(
                PointDict, index_max, 'incAngle', 'height', 'x', 'y', 'longitude', 'latitude')

            if save_peak:
                # 保存 DEM极值点
                # for x, y in zip(Lon_max, Lat_max):
                #     PeakPoint_list.append(ee.Feature(ee.Geometry.Point(x, y), {'values': 1}))
                if 'PeakFeatureCollection' in locals():
                    PeakFeatureCollection = PeakFeatureCollection.merge(ee.FeatureCollection([ee.Feature(
                        ee.Geometry.Point(x, y), {'values': 1}) for x, y in zip(Lon_max, Lat_max)]))
                else:
                    PeakFeatureCollection = ee.FeatureCollection([ee.Feature(
                        ee.Geometry.Point(x, y), {'values': 1}) for x, y in zip(Lon_max, Lat_max)])

            # --------检索出一条线上所有Peak点的畸变
            for each in range(len(index_max)):

                rx = X_max[each]
                ry = Y_max[each]
                rh = Z_max[each]
                rlon = Lon_max[each]
                rlat = Lat_max[each]
                rindex = index_max[each]
                r_angle = Angle_max[each]

                Pixels_cal = 900 // scale

                # 被动叠掩（仅山底）
                if index_max[each] - Pixels_cal > 0:
                    rangeIndex = range(rindex - Pixels_cal, rindex)
                else:
                    rangeIndex = range(0, rindex)

                ## 计算坡度梯度会少一位，因此在第一位进行补‍0，保持元素数量一致
                PointDict['Grad_alpha_rRad'] = np.insert(np.diff(PointDict['alpha_rRad']), 0, 0)
                L_h, L_x, L_y, L_lon, L_lat, L_angle, L_Grad_alpha_rRad = EasyIndex(
                    PointDict, rangeIndex, 'height', 'x', 'y', 'longitude', 'latitude', 'incAngle', 'Grad_alpha_rRad')

                Llay_angle_iRad = np.arctan((rh - L_h) / np.sqrt(np.square(L_x - rx) + np.square(L_y - ry)))
                Llay_angle = Llay_angle_iRad * 180 / math.pi

                index_Llay = np.where(Llay_angle > r_angle)[0]

                if len(index_Llay) != 0:
                    if line_points_connect:
                        index_Llay = range(np.where(Llay_angle > r_angle)[0][0], len(Llay_angle))
                    tlon_Llay = L_lon[index_Llay]
                    tlat_Llay = L_lat[index_Llay]
                    if Peak_Llay:
                        tlon_Llay = np.append(tlon_Llay, rlon)
                        tlat_Llay = np.append(tlat_Llay, rlat)

                    for i,j in zip(tlon_Llay,tlat_Llay):
                        if [i,j] not in LPassive_layover:
                            LPassive_layover.append([i,j])

                # 阴影
                if index_max[each] + Pixels_cal < len(PointDict['x']):
                    rangeIndex = range(index_max[each] + 1, index_max[each] + Pixels_cal)
                else:
                    rangeIndex = range(index_max[each] + 1, len(PointDict['x']))

                R_h, R_x, R_y, R_lon, R_lat = EasyIndex(
                    PointDict, rangeIndex, 'height', 'x', 'y', 'longitude', 'latitude')
                R_angle_iRad = np.arctan(
                    (rh - R_h) / np.sqrt(np.square(R_x - rx) + np.square(R_y - ry)) + sys.float_info.min)
                R_angle = R_angle_iRad * 180 / math.pi
                index_Shadow = np.where(R_angle > (90 - r_angle))[0]

                if len(index_Shadow) != 0:
                    if line_points_connect:
                        index_Shadow = range(0, np.where(R_angle > (90 - r_angle))[0][-1])
                    tlon_Shadow = R_lon[index_Shadow]
                    tlat_Shadow = R_lat[index_Shadow]
                    if Peak_shdow:
                        tlon_Shadow = np.append(tlon_Shadow, rlon)
                        tlat_Shadow = np.append(tlat_Shadow, rlat)

                    for i,j in zip(tlon_Shadow,tlat_Shadow):
                        if [i,j] not in Passive_shadow:
                            Passive_shadow.append([i,j])

                if len(index_Llay) != 0:
                    # 被动叠掩(山顶右侧),与阴影运算交叉
                    # 求取在左侧叠掩范围内变化最大值点,在index_Llay范围内求最大坡度变化值
                    try:
                        L_bottom = index_Llay[np.argmax(L_Grad_alpha_rRad[index_Llay])]
                        # layoverM_x, layoverM_y = L_x[L_bottom], L_y[L_bottom],  # ,L_lon[L_bottom], L_lat[L_bottom]
                        layoverM_x, layoverM_y = rx, ry
                        layoverM_h = L_h[L_bottom]  # np.min(L_h[index_Llay])
                        layoverM_angle = r_angle
                    except:
                        raise IndexError('索引问题each={}'.format(each))

                    Rlay_angle_iRad = np.arctan(
                        (R_h - layoverM_h) / np.sqrt(np.square(R_x - layoverM_x) + np.square(R_y - layoverM_y)))
                    Rlay_angle = Rlay_angle_iRad * 180 / math.pi
                    index_Rlayover = np.where(Rlay_angle > layoverM_angle)[0]

                    if len(index_Rlayover) != 0:
                        if line_points_connect:
                            index_Rlayover = range(0, np.where(Rlay_angle > layoverM_angle)[0][-1])
                        tlon_RLay = R_lon[index_Rlayover]
                        tlat_RLay = R_lat[index_Rlayover]
                        if Peak_Rlay:
                            tlon_RLay = np.append(tlon_RLay, rlon)
                            tlat_RLay = np.append(tlat_RLay, rlat)

                        for i,j in zip(tlon_RLay,tlat_RLay):
                            if [i,j] not in RPassive_layover:
                                RPassive_layover.append([i,j])


    LeftLayover_PointFeatures= ee.FeatureCollection([ee.Feature(ee.Geometry.Point(x, y), {'values': 1}) for x, y in LPassive_layover])
    RightLayover_PointFeatures = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(x, y), {'values': 3}) for x, y in RPassive_layover])
    Shadow_PointFeatures = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(x, y), {'values': 5}) for x, y in Passive_shadow])

    LeftLayover=ee.Image().paint(LeftLayover_PointFeatures,'values').clip(AOI_buffer).reproject(crs=proj, scale=scale)
    RightLayover=ee.Image().paint(RightLayover_PointFeatures,'values').clip(AOI_buffer).reproject(crs=proj, scale=scale)
    Shadow=ee.Image().paint(Shadow_PointFeatures,'values').clip(AOI_buffer).reproject(crs=proj, scale=scale)

    if save_peak:
        return LeftLayover.toInt8(), RightLayover.toInt8(), Shadow.toInt8(), PeakFeatureCollection
    else:
        return LeftLayover.toInt8(), RightLayover.toInt8(), Shadow.toInt8()
    
# 被动几何畸变识别（方法2）
def Line_Correct2(cal_image,AOI_buffer,Templist, orbitProperties_pass, proj, scale:int, cal_image_scale: int):
    '''
    Watch Out!
    这个方法结果仅仅返回了被动叠掩和被动阴影，需要与主动叠掩与主动阴影结合
    '''
    line_points_list = []
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
                                         cal_image_scale, PointDict['y'] * cal_image_scale  # 像素行列10m分辨率，由proj得
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
                        (r_Range_h - s_h[i]) / (np.sqrt(np.square(r_Range_x - s_x[i]) + np.square(r_Range_y - s_y[i])))+  sys.float_info.min)
                    Rlay_end_iRad = np.arctan(
                        (r_Range_h - d_h[i]) / (np.sqrt(np.square(r_Range_x - d_x[i]) + np.square(r_Range_y - d_y[i]))) +  sys.float_info.min)
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
        [ee.Feature(ee.Geometry.Point(x, y), {'values': 3}) for x, y in zip(r_lon_sum, r_lat_sum)])
    shadow_featurecollection = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point(x, y), {'values': 5}) for x, y in zip(sh_lon_sum, sh_lat_sum)])
    llay_featurecollection = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point(x, y), {'values': 2}) for x, y in zip(l_lon_sum, l_lat_sum)])
    image_rlayover=ee.Image().paint(rlay_featurecollection,'values')
    image_llayover=ee.Image().paint(llay_featurecollection,'values')
    image_shadow=ee.Image().paint(shadow_featurecollection,'values')
    img_rlayover=image_rlayover.clip(AOI_buffer).reproject(crs=proj, scale=scale)
    img_llayover=image_llayover.clip(AOI_buffer).reproject(crs=proj, scale=scale)
    img_shadow=image_shadow.clip(AOI_buffer).reproject(crs=proj, scale=scale)
    return img_llayover.toInt8(),img_rlayover.toInt8(),img_shadow.toInt8()

