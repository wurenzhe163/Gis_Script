from functools import partial
import ee,os,sys
import numpy as np
import math
from tqdm import tqdm
from scipy.signal import argrelextrema

# Add the parent directory to the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the same directory
from .GEE_CorreterAndFilters import ImageFilter, S1Corrector
from .GEE_DataIOTrans import DataTrans, BandTrans
from .GEEMath import angle2slope, time_difference

# ---------------------------------S1几何畸变检测、冰湖提取专用
def load_S1collection(aoi, start_date, end_date, middle_date, Filter=None, FilterSize=30):

    '''s1数据加载'''
    s1_col = (ee.ImageCollection("COPERNICUS/S1_GRD")
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filterBounds(aoi)
                .filterDate(start_date, end_date))

    # 裁剪并计算空洞数量
    s1_col = s1_col.map(partial(DataTrans.rm_nodata, AOI=aoi))

    # 图像滤波，可选
    if Filter:
        print('Begin Filter ...')
        if Filter == 'leesigma':
            s1_col = s1_col.map(ImageFilter.leesigma(FilterSize))
        elif Filter == 'RefinedLee':
            s1_col = s1_col.map(ImageFilter.RefinedLee)
        elif Filter == 'gammamap':
            s1_col = s1_col.map(ImageFilter.gammamap(FilterSize))
        elif Filter == 'boxcar':
            s1_col = s1_col.map(ImageFilter.boxcar(FilterSize))
        else:
            print('Wrong Filter')
    else:
        print('Without Filter')

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
        s1_ascending.median().reproject(s1_ascending.first().select(0).projection().crs(), None, 10).set({'synthesis': 1}),
        filtered_collection_A.filter(ee.Filter.eq('time_difference',filtered_collection_A.aggregate_min('time_difference'))).first().set({'synthesis': 0})
                                    )

    filtered_collection_D = s1_descending.filter(ee.Filter.eq('numNodata', 0))
    has_images_without_nodata_D = filtered_collection_D.size().eq(0)
    s1_descending = ee.Algorithms.If(
        has_images_without_nodata_D,
        s1_descending.median().reproject(s1_descending.first().select(0).projection().crs(), None, 10).set({'synthesis': 1}),
        filtered_collection_D.filter(ee.Filter.eq('time_difference',filtered_collection_D.aggregate_min('time_difference'))).first().set({'synthesis': 0})
                                        )
    return ee.Image(s1_ascending), ee.Image(s1_descending)  # ,s1_col_copy

class DEM_caculator(object):
    def slop_aspect(elevation, proj, geom, phi_iRad):
        """
        phi_iRad ：方位角（弧度）
        alpha_sRad ：坡度(与地面夹角)
        phi_sRad ：坡向角，(坡度陡峭度)坡与正北方向夹角(陡峭度)，从正北方向起算，顺时针计算角度
        alpha_rRad ：距离向分解
        alpha_azRad ：方位向分解

        phi_rRad：(飞行方向角度-坡度陡峭度)飞行方向与坡向之间的夹角
        """
        alpha_sRad = ee.Terrain.slope(elevation).select('slope').multiply(
            np.pi / 180).setDefaultProjection(proj).clip(geom)  # 坡度(与地面夹角)
        phi_sRad = ee.Terrain.aspect(elevation).select('aspect').multiply(
            np.pi / 180).setDefaultProjection(proj).clip(geom)  # 坡向角，(坡度陡峭度)坡与正北方向夹角(陡峭度)，从正北方向起算，顺时针计算角度
        phi_rRad = phi_iRad.subtract(phi_sRad)  # (飞行方向角度-坡度陡峭度)飞行方向与坡向之间的夹角

        # 分解坡度，在水平方向和垂直方向进行分解，为固定公式，cos对应水平分解，sin对应垂直分解
        alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()  # 距离向分解
        alpha_azRad = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()  # 方位向分解
        return alpha_sRad, phi_sRad, alpha_rRad, alpha_azRad

    def Origin_s1_slope_correction(collection, elevation, model, buffer=0):
        '''This function applies the slope correction on a collection of Sentinel-1 data

           :param collection: ee.Collection or ee.Image of Sentinel-1
           :param elevation: ee.Image of DEM
           :param model: model to be applied (volume/surface)
           :param buffer: buffer in meters for layover/shadow amsk
           #
           :returns: ee.Image
        '''

        def _erode(image, distance):
            '''Buffer function for raster
                腐蚀算法，输入的图像需要额外的缓冲
            :param image: ee.Image that shoudl be buffered
            :param distance: distance of buffer in meters

            :returns: ee.Image
            '''

            d = (image.Not().unmask(1).fastDistanceTransform(30).sqrt().multiply(
                ee.Image.pixelArea().sqrt()))

            return image.updateMask(d.gt(distance))

        def _masking(alpha_rRad, theta_iRad, buffer):
            '''Masking of layover and shadow
                获取几何畸变区域
            :param alpha_rRad: ee.Image of slope steepness in range
            :param theta_iRad: ee.Image of incidence angle in radians
            :param buffer: buffer in meters

            :returns: ee.Image
            '''
            # layover, where slope > radar viewing angle
            layover = alpha_rRad.lt(theta_iRad).rename('layover')

            # shadow
            ninetyRad = ee.Image.constant(90).multiply(np.pi / 180)
            shadow = alpha_rRad.gt(
                ee.Image.constant(-1).multiply(
                    ninetyRad.subtract(theta_iRad))).rename('shadow')

            # add buffer to layover and shadow
            if buffer > 0:
                layover = _erode(layover, buffer)
                shadow = _erode(shadow, buffer)

            # combine layover and shadow
            no_data_mask = layover.And(shadow).rename('no_data_mask')

            return layover.addBands(shadow).addBands(no_data_mask)

        def _correct(image):
            '''This function applies the slope correction and adds layover and shadow masks

            '''

            # get the image geometry and projection
            geom = image.geometry()
            proj = image.select(1).projection()

            # calculate the look direction
            heading = (ee.Terrain.aspect(image.select('angle')).reduceRegion(
                ee.Reducer.mean(), geom, 1000).get('aspect'))

            # Sigma0 to Power of input image，Sigma0是指雷达回波信号的强度
            sigma0Pow = ee.Image.constant(10).pow(image.divide(10.0))

            # the numbering follows the article chapters
            # 2.1.1 Radar geometry
            theta_iRad = image.select('angle').multiply(np.pi / 180)
            phi_iRad = ee.Image.constant(heading).multiply(np.pi / 180)

            # 2.1.2 Terrain geometry
            alpha_sRad = ee.Terrain.slope(elevation).select('slope').multiply(
                np.pi / 180).setDefaultProjection(proj).clip(geom)
            phi_sRad = ee.Terrain.aspect(elevation).select('aspect').multiply(
                np.pi / 180).setDefaultProjection(proj).clip(geom)

            # we get the height, for export
            height = elevation.setDefaultProjection(proj).clip(geom)

            # 2.1.3 Model geometry
            # reduce to 3 angle
            phi_rRad = phi_iRad.subtract(phi_sRad)

            # slope steepness in range (eq. 2)
            alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()

            # slope steepness in azimuth (eq 3)
            alpha_azRad = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()

            # local incidence angle (eq. 4)
            theta_liaRad = (alpha_azRad.cos().multiply(
                (theta_iRad.subtract(alpha_rRad)).cos())).acos()
            theta_liaDeg = theta_liaRad.multiply(180 / np.pi)

            # 2.2
            # Gamma_nought
            gamma0 = sigma0Pow.divide(theta_iRad.cos())
            gamma0dB = ee.Image.constant(10).multiply(gamma0.log10()).select(
                ['VV', 'VH'], ['VV_gamma0', 'VH_gamma0'])
            ratio_gamma = (gamma0dB.select('VV_gamma0').subtract(
                gamma0dB.select('VH_gamma0')).rename('ratio_gamma0'))

            if model == 'volume':
                scf = S1Corrector.volumetric_model_SCF(theta_iRad, alpha_rRad)

            if model == 'surface':
                scf = S1Corrector.surface_model_SCF(theta_iRad, alpha_rRad, alpha_azRad)

            # apply model for Gamm0_f
            gamma0_flat = gamma0.divide(scf)
            gamma0_flatDB = (ee.Image.constant(10).multiply(
                gamma0_flat.log10()).select(['VV', 'VH'],['VV_gamma0flat', 'VH_gamma0flat']))
            masks = _masking(alpha_rRad, theta_iRad, buffer)

            # calculate the ratio for RGB vis
            ratio_flat = (gamma0_flatDB.select('VV_gamma0flat').subtract(
                gamma0_flatDB.select('VH_gamma0flat')).rename('ratio_gamma0flat'))

            return (image.rename(['VV_sigma0', 'VH_sigma0', 'incAngle'])
                    .addBands(gamma0dB)
                    .addBands(ratio_gamma)
                    .addBands(gamma0_flatDB)
                    .addBands(ratio_flat)
                    .addBands(alpha_rRad.rename('alpha_rRad'))
                    .addBands(alpha_azRad.rename('alpha_azRad'))
                    .addBands(phi_sRad.rename('aspect'))
                    .addBands(alpha_sRad.rename('slope'))
                    .addBands(theta_iRad.rename('theta_iRad'))
                    .addBands(theta_liaRad.rename('theta_liaRad'))
                    .addBands(masks).addBands(height.rename('elevation')))

        # run and return correction
        if type(collection) == ee.imagecollection.ImageCollection:
            return collection.map(_correct)
        elif type(collection) == ee.image.Image:
            return _correct(collection)
        else:
            print('Check input type, only image collection or image can be input')
    
    def My_s1_slope_correction(s1_ascending, s1_descending,
                            AOI_buffer, DEM,
                            model, Origin_scale=10,
                            DistorMethed='RS'):
        volumetric_dict = {}
        for image, orbitProperties_pass in zip([s1_ascending, s1_descending], ['ASCENDING', 'DESCENDING']):
            # 获取投影几何
            geom = image.geometry()
            proj = image.select(0).projection()

            # 计算方位向、距离向以及图像的四个角点、构筑计算辅助线
            azimuthEdge, rotationFromNorth, startpoint, endpoint, coordinates_dict = S1Corrector.getS1Corners(image, AOI_buffer,orbitProperties_pass)
            Heading = azimuthEdge.get('azimuth')
            ## 舍弃方法，添加rotationFromNorth对原有的s1_azimuth_across进行修正
            # Angle_aspect = ee.Terrain.aspect(image.select('angle'))
            # s1_azimuth_across = Angle_aspect.reduceRegion(ee.Reducer.mean(), geom, 1000).get('aspect')
            # s1_azimuth_across = ee.Number(s1_azimuth_across).add(rotationFromNorth)
            s1_azimuth_across = ee.Number(Heading).subtract(90.0) # 距离向
            theta_iRad = image.select('angle').multiply(np.pi / 180)  # 地面入射角度转为弧度
            phi_iRad = ee.Image.constant(s1_azimuth_across).multiply(np.pi / 180)  # 距离向转弧度

            #计算地形几何信息
            alpha_sRad, phi_sRad, alpha_rRad, alpha_azRad = DEM_caculator.slop_aspect(DEM, proj, geom, phi_iRad)
            theta_liaRad = (alpha_azRad.cos().multiply((theta_iRad.subtract(alpha_rRad)).cos())).acos()  # LIA
            theta_liaDeg = theta_liaRad.multiply(180 / np.pi)  # LIA转弧度

            # 根据入射角度修订
            sigma0Pow = ee.Image.constant(10).pow(image.divide(10.0))
            gamma0 = sigma0Pow.divide(theta_iRad.cos())
            gamma0dB = ee.Image.constant(10).multiply(gamma0.log10()).select(['VV', 'VH'],
                                                                             ['VV_gamma0', 'VH_gamma0'])  # 根据角度修订入射值
            
            # 判断几何畸变
            if DistorMethed == 'RS':
                # ------------------------------RS几何畸变区域--------------------------------- 同戴可人：高山峡谷区滑坡灾害隐患InSAR早期识别
                layover = alpha_rRad.gt(theta_iRad).rename('layover')
                ninetyRad = ee.Image.constant(90).multiply(np.pi / 180)
                shadow = alpha_rRad.lt(ee.Image.constant(-1).multiply(ninetyRad.subtract(theta_iRad))).rename('shadow')
            elif DistorMethed == 'IJRS':
                # ------------------------------IJRS几何畸变区域-------------------------------
                layover = alpha_rRad.gt(theta_iRad).rename('layover')
                shadow = theta_liaDeg.gt(ee.Image.constant(85).multiply(np.pi / 180)).rename('shadow')
            elif DistorMethed == 'Wu':
                # ------------------------------武大学报几何畸变区域---------------------------
                layover = theta_liaDeg.lt(ee.Image.constant(0).multiply(np.pi / 180)).rename('layover')
                shadow = theta_liaDeg.gt(ee.Image.constant(90).multiply(np.pi / 180)).rename('shadow')
            # RINDEX，暂时无用
            else:
                raise Exception('DistorMethed is not right!')

            # combine layover and shadow,因为shadow和layover都是0
            # no_data_maskrgb = BandTrans.add_DistorRgbmask(image, layover=layover, shadow=shadow)
            slop_correction = S1Corrector.volumetric(model, theta_iRad, alpha_rRad, alpha_azRad,gamma0)
            Eq_pixels = DataTrans.Eq_pixels
            image2 = ( Eq_pixels(image.select('VV')).rename('VV_sigma0')
                      .addBands(Eq_pixels(image.select('VH')).rename('VH_sigma0'))
                      .addBands(Eq_pixels(image.select('angle')).rename('incAngle'))
                      .addBands(Eq_pixels(alpha_sRad.reproject(crs=proj, scale=Origin_scale)).rename('alpha_sRad'))
                      .addBands(Eq_pixels(alpha_rRad.reproject(crs=proj, scale=Origin_scale)).rename('alpha_rRad'))
                      .addBands(gamma0dB)
                      .addBands(Eq_pixels(slop_correction['gamma0_flat'].select('VV')).rename('VV_gamma0_flat'))
                      .addBands(Eq_pixels(slop_correction['gamma0_flat'].select('VH')).rename('VH_gamma0_flat'))
                      .addBands(Eq_pixels(slop_correction['gamma0_flatDB'].select('VV_gamma0flat')).rename('VV_gamma0_flatDB'))
                      .addBands(Eq_pixels(slop_correction['gamma0_flatDB'].select('VH_gamma0flat')).rename('VH_gamma0_flatDB'))
                      .addBands(Eq_pixels(layover).rename('layover'))
                      .addBands(Eq_pixels(shadow).rename('shadow'))
                    #   .addBands(no_data_maskrgb)
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

class S1_CalDistor(object):
    '''
    S1几何畸变检测专用函数，根据模拟的入射线，迭代计算几何畸变区与
    '''
    # 一个专用赋值函数，用于从Data中提取需要的数据
    EasyIndex = lambda Data, Index, *Keys: [Data[key][Index] for key in Keys]
    # 将图像转为点数据表达
    def reduce_tolist(each, scale):
        return ee.Image(each).reduceRegion(
            reducer=ee.Reducer.toList(), geometry=each.geometry(), scale=scale, maxPixels=1e13)
            
    def Line2Points(feature,region,scale=30):
        # 从Feature中提取线几何对象
        line_geometry = ee.Feature(feature).geometry().intersection(region, maxError=1)
        
        # 获取线的所有坐标点
        coordinates = line_geometry.coordinates()
        start_point = ee.List(coordinates.get(0)) # 起点坐标
        end_point = ee.List(coordinates.get(-1))  # 终点坐标
        
        # 获取线段的总长度
        length = line_geometry.length()

        # 计算插入点的数量
        num_points = length.divide(scale).subtract(1).floor()

        # 计算每个间隔点的坐标
        def interpolate(i):
            i = ee.Number(i)
            fraction = i.divide(num_points)
            interpolated_lon = ee.Number(start_point.get(0)).add(
                ee.Number(end_point.get(0)).subtract(ee.Number(start_point.get(0))).multiply(fraction))
            interpolated_lat = ee.Number(start_point.get(1)).add(
                ee.Number(end_point.get(1)).subtract(ee.Number(start_point.get(1))).multiply(fraction))
            return ee.Feature(ee.Geometry.Point([interpolated_lon, interpolated_lat]))

        # 使用条件表达式过滤长度为0的线段
        filtered_points = ee.FeatureCollection(ee.Algorithms.If(num_points.gt(0), 
                                                                ee.FeatureCollection(ee.List.sequence(1, num_points).map(interpolate)),
                                                                ee.FeatureCollection([])))

        return filtered_points
    
    def AuxiliaryLine2Point_numpy(cal_image, s1_azimuth_across, coordinates_dict, Auxiliarylines, scale):
        '''获取所有待计算的计算线,线是矢量'''
        # 计算斜率
        K = angle2slope(s1_azimuth_across).getInfo()
        
        # 过Auxiliarylines中的点，从最小经度到最大经度
        Max_Lon = coordinates_dict['maxLon'].getInfo()
        Min_Lon = coordinates_dict['minLon'].getInfo()

        AuxiliaryPoints = S1_CalDistor.reduce_tolist(cal_image.select(['longitude', 'latitude']).clip(Auxiliarylines),scale=scale).getInfo()
        
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
    
    def AuxiliaryLine2Point_old(cal_image, s1_azimuth_across, coordinates_dict, Auxiliarylines, scale):
        '''获取所有待计算的计算线,线是矢量，全在GEE服务器端处理
        由于采用图像处理，出现了一些由于像素偏移导致的误差'''
        # 直接将角度转换为斜率，这里假设angle2slope函数返回的是ee.Number类型
        K = angle2slope(s1_azimuth_across) 
        
        Max_Lon = coordinates_dict['maxLon']
        Min_Lon = coordinates_dict['minLon']

        # 使用map函数代替循环处理辅助线上的所有点数据
        def create_line(coords):
            lon = ee.Number(coords.get(0))
            lat = ee.Number(coords.get(1))
            C = lat.subtract(K.multiply(lon))
            Min_Lon_Y = K.multiply(Min_Lon).add(C)
            Max_Lon_Y = K.multiply(Max_Lon).add(C)
            line = ee.Geometry.LineString([[Min_Lon, Min_Lon_Y], [Max_Lon, Max_Lon_Y]])
            return ee.Feature(line)
        
        # 将图像裁剪到辅助线范围并降采样为点集
        points = S1_CalDistor.reduce_tolist(cal_image.select(['longitude', 'latitude']).clip(Auxiliarylines),scale=scale)

        lon_list = ee.List(points.get('longitude'))
        lat_list = ee.List(points.get('latitude'))

        # Zip the lists and map the function over the zipped list
        coords_list = lon_list.zip(lat_list)
        lines = coords_list.map(lambda coords: create_line(ee.List(coords)))

        return lines
    
    def AuxiliaryLine2Point(s1_azimuth_across, coordinates_dict, Auxiliarylines, region,scale):
        '''获取所有待计算的计算线,线是矢量，全在GEE服务器端处理'''

        # 直接将角度转换为斜率，这里假设angle2slope函数返回的是ee.Number类型
        K = angle2slope(s1_azimuth_across) 
        
        Max_Lon = coordinates_dict['maxLon']
        Min_Lon = coordinates_dict['minLon']

        # 使用map函数代替循环处理辅助线上的所有点数据
        def create_line(coords):
            lon = ee.Number(coords.get(0))
            lat = ee.Number(coords.get(1))
            C = lat.subtract(K.multiply(lon))
            Min_Lon_Y = K.multiply(Min_Lon).add(C)
            Max_Lon_Y = K.multiply(Max_Lon).add(C)
            line = ee.Geometry.LineString([[Min_Lon, Min_Lon_Y], [Max_Lon, Max_Lon_Y]])
            return ee.Feature(line)
        
        # 将图像裁剪到辅助线范围并降采样为点集
        points = S1_CalDistor.Line2Points(Auxiliarylines,region,scale=scale)
        
        list_of_dicts = points.geometry().coordinates()
        lon_list = list_of_dicts.map(lambda x: ee.List(x).get(0))
        lat_list = list_of_dicts.map(lambda x: ee.List(x).get(1))

        # Zip the lists and map the function over the zipped list
        coords_list = lon_list.zip(lat_list)
        lines = coords_list.map(lambda coords: create_line(ee.List(coords)))

        return lines

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
                Angle_max, Z_max, X_max, Y_max, Lon_max, Lat_max = S1_CalDistor.EasyIndex(
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
                    L_h, L_x, L_y, L_lon, L_lat, L_angle, L_Grad_alpha_rRad = S1_CalDistor.EasyIndex(
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
                            # 使用列表收集数据，然后一次性转换为numpy数组，避免重复的np.append
                            tlon_Llay_list = list(tlon_Llay)
                            tlat_Llay_list = list(tlat_Llay)
                            tlon_Llay_list.append(rlon)
                            tlat_Llay_list.append(rlat)
                            tlon_Llay = np.array(tlon_Llay_list)
                            tlat_Llay = np.array(tlat_Llay_list)

                        for i, j in zip(tlon_Llay, tlat_Llay):
                            if [i, j] not in LPassive_layover:
                                LPassive_layover.append([i, j])

                    # 阴影
                    if index_max[each] + Pixels_cal < len(PointDict['x']):
                        rangeIndex = range(index_max[each] + 1, index_max[each] + Pixels_cal)
                    else:
                        rangeIndex = range(index_max[each] + 1, len(PointDict['x']))

                    R_h, R_x, R_y, R_lon, R_lat = S1_CalDistor.EasyIndex(
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
                            # 使用列表收集数据，然后一次性转换为numpy数组，避免重复的np.append
                            tlon_Shadow_list = list(tlon_Shadow)
                            tlat_Shadow_list = list(tlat_Shadow)
                            tlon_Shadow_list.append(rlon)
                            tlat_Shadow_list.append(rlat)
                            tlon_Shadow = np.array(tlon_Shadow_list)
                            tlat_Shadow = np.array(tlat_Shadow_list)

                        for i, j in zip(tlon_Shadow, tlat_Shadow):
                            if [i, j] not in Passive_shadow:
                                Passive_shadow.append([i, j])

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

                            for i, j in zip(tlon_RLay, tlat_RLay):
                                if [i, j] not in RPassive_layover:
                                    RPassive_layover.append([i, j])

        LeftLayover_PointFeatures = ee.FeatureCollection(
            [ee.Feature(ee.Geometry.Point(x, y), {'values': 1}) for x, y in LPassive_layover])
        RightLayover_PointFeatures = ee.FeatureCollection(
            [ee.Feature(ee.Geometry.Point(x, y), {'values': 3}) for x, y in RPassive_layover])
        Shadow_PointFeatures = ee.FeatureCollection(
            [ee.Feature(ee.Geometry.Point(x, y), {'values': 5}) for x, y in Passive_shadow])

        LeftLayover = ee.Image().paint(LeftLayover_PointFeatures, 'values').clip(AOI_buffer).reproject(crs=proj,
                                                                                                       scale=scale)
        RightLayover = ee.Image().paint(RightLayover_PointFeatures, 'values').clip(AOI_buffer).reproject(crs=proj,
                                                                                                         scale=scale)
        Shadow = ee.Image().paint(Shadow_PointFeatures, 'values').clip(AOI_buffer).reproject(crs=proj, scale=scale)

        if save_peak:
            return LeftLayover.toInt8(), RightLayover.toInt8(), Shadow.toInt8(), PeakFeatureCollection
        else:
            return LeftLayover.toInt8(), RightLayover.toInt8(), Shadow.toInt8()

    # 被动几何畸变识别（方法2）
    def Line_Correct2(cal_image, AOI_buffer, Templist, orbitProperties_pass, proj, scale: int, cal_image_scale: int):
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
                            (r_Range_h - s_h[i]) / (np.sqrt(
                                np.square(r_Range_x - s_x[i]) + np.square(r_Range_y - s_y[i]))) + sys.float_info.min)
                        Rlay_end_iRad = np.arctan(
                            (r_Range_h - d_h[i]) / (np.sqrt(
                                np.square(r_Range_x - d_x[i]) + np.square(r_Range_y - d_y[i]))) + sys.float_info.min)
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
        image_rlayover = ee.Image().paint(rlay_featurecollection, 'values')
        image_llayover = ee.Image().paint(llay_featurecollection, 'values')
        image_shadow = ee.Image().paint(shadow_featurecollection, 'values')
        img_rlayover = image_rlayover.clip(AOI_buffer).reproject(crs=proj, scale=scale)
        img_llayover = image_llayover.clip(AOI_buffer).reproject(crs=proj, scale=scale)
        img_shadow = image_shadow.clip(AOI_buffer).reproject(crs=proj, scale=scale)
        return img_llayover.toInt8(), img_rlayover.toInt8(), img_shadow.toInt8()

