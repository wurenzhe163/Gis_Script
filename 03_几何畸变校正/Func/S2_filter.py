# CLOUD_FILTER = 60                  # 过滤s2 大于指定云量的数据
# CLD_PRB_THRESH = 15                # s2cloudless 概率值阈值[0-100],原实验是50
# NIR_DRK_THRESH = 0.15              # 非水暗像素判断阈值
# CLD_PRJ_DIST = 1                   # 根据 CLD_PRJ_DIST 输入指定的距离从云中投射阴影
# BUFFER = 50                        # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input
import ee
from functools import partial
def clip_AOI(col, AOI): return col.clip(AOI)
def add_cloud_bands(img,CLD_PRB_THRESH):
    """Define a function to add the s2cloudless probability layer
    and derived cloud mask as bands to an S2 SR image input."""
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def add_shadow_bands(img,NIR_DRK_THRESH,CLD_PRJ_DIST):
    """Define a function to add dark pixels,
    cloud projection, and identified shadows as bands to an S2 SR image input.
    Note that the image input needs to be the result of the above add_cloud_bands function
    because it relies on knowing which pixels are considered cloudy ('clouds' band)."""
    # 从 SCL 波段识别水像素, 仅适用于L2A，采用L1C计算MNDWI
    not_water = img.select('SCL').neq(6)

    # 识别非水的暗 NIR 像素(潜在的云阴影像素)。.
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # 确定云投射云影的方向(假设是 UTM 投影)。
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # 根据 CLD_PRJ_DIST 输入指定的距离从云中投射阴影
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')
    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cld_shdw_mask(img,BUFFER,CLD_PRJ_DIST,NIR_DRK_THRESH,CLD_PRB_THRESH):
    """
    添加cloud和shadow mask
    """
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img,CLD_PRB_THRESH)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud,NIR_DRK_THRESH=NIR_DRK_THRESH,CLD_PRJ_DIST=CLD_PRJ_DIST)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)

def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()
    # Subset reflectance bands and update their masks, return the result.
    return img.select(['B.*','clouds','dark_pixels','shadows','cloudmask']).updateMask(not_cld_shdw)

def merge_s2_collection(aoi, start_date, end_date,CLOUD_FILTER,BUFFER,CLD_PRJ_DIST,CLD_PRB_THRESH,NIR_DRK_THRESH):
    """筛选S2图像以及S2_cloud图像，并将两个collection连接"""
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)).map(partial(clip_AOI,AOI=aoi)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date).map(partial(clip_AOI,AOI=aoi)))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    # 固定用法，将两个collection通过属性值连接起来，s2cloudless整体作为一个属性写入
    s2_sr_cld_col = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'})
    }))

    s2_sr_cld_col_disp = s2_sr_cld_col.map(partial(add_cld_shdw_mask,BUFFER=BUFFER,
                            CLD_PRJ_DIST=CLD_PRJ_DIST,
                            CLD_PRB_THRESH=CLD_PRB_THRESH,
                            NIR_DRK_THRESH=NIR_DRK_THRESH))
    # 使用clip赋予geometry
    s2_sr_median = s2_sr_cld_col_disp.map(apply_cld_shdw_mask).median().clip(aoi).int16()
    return s2_sr_median