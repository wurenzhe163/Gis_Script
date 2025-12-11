import ee
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .GEE_DataIOTrans import DataIO
from PackageDeepLearn.utils.DataIOTrans import make_dir

# --获取子数据集(主要用于删除GEE Asset)

class GEE_Asset(object):
    # 获取指定父文件夹或图像集合下的所有子资产（包括文件夹和图像集合），并返回它们的名称列表
    def get_asset_list(parent):
        parent_asset = ee.data.getAsset(parent)
        parent_id = parent_asset['name']
        parent_type = parent_asset['type']
        asset_list = []
        child_assets = ee.data.listAssets({'parent': parent_id})['assets']
        for child_asset in child_assets:
            child_id = child_asset['name']
            child_type = child_asset['type']
            if child_type in ['FOLDER', 'IMAGE_COLLECTION']:
                # Recursively call the function to get child assets
                asset_list.extend(GEE_Asset.get_asset_list(child_id))
            else:
                asset_list.append(child_id)
        return asset_list

    # 删除数据集，包含子数据集（慎用）
    def delete_asset_list(asset_list, save_parent=1):
        for asset in asset_list:
            ee.data.deleteAsset(asset)
        if save_parent:
            print('删除成功,save parent')
        else:
            ee.data.deleteAsset(os.path.dirname(asset_list[0]))
            print('删除成功,同步删除 parent')

    try:
        from geeup import geeup
        # 重新载入变量，通过将数据上传至GEE再重新载入，打断惰性运算的算力要求
        def reload_variable(region,
                            scale=10,
                            save_dir='./test',
                            dest='users/mrwurenzhe/test',
                            metaName='test.csv',
                            eeUser='mrwurenzhe@furwas.com',
                            overwrite='yes',
                            delgeeUp=False,
                            **parms):
            # 转为绝对路径，创建文件夹
            if not os.path.isabs(save_dir):
                save_dir = os.path.join(os.getcwd(), os.path.basename(save_dir))
            save_dir = make_dir(save_dir)
            meta_csv = os.path.join(save_dir, metaName)
            # 导出
            for key, value in parms.items():
                fileName = os.path.join(save_dir, key + '.tif')
                DataIO.Geemap_export(fileName, image=value, region=region, scale=scale)

            # 生成meta
            geeup.getmeta(save_dir, meta_csv)  # !geeup getmeta --input save_dir --metadata meta_csv
            # 上传
            geeup.upload(user=eeUser, source_path=save_dir, pyramiding="MEAN", mask=False, nodata_value=None,
                         metadata_path=meta_csv, destination_path=dest,
                         overwrite=overwrite)  # !geeup upload --source save_dir --dest dest -m meta_csv --nodata 0 -u eeUser
            Images = ee.ImageCollection(dest)
            # 重新导入
            for key in parms.keys():
                vars()[key] = Images.filter(ee.Filter.eq('id_no', ee.String(key))).first()
                parms[key] = vars()[key]
            print('采用上传替换数据，打断惰性运算的算力需求')

            if delgeeUp:
                asset_list = GEE_Asset.get_asset_list(dest)
                GEE_Asset.delete_asset_list(asset_list, save_parent=0)
            return parms
    except:
        print('geeup not import')
class S1_Cheker(object):
    def CheckDuplicateBands(s1_image):
        '''检查波段是否完整(同时具备'VV'\'VH'),并修复'''
        # 获取图像的波段名称列表
        band_names = s1_image.bandNames()

        # 创建一个条件表达式来检查'VV'和'VH'波段是否存在，并相应地处理
        def add_missing_band(image, existing_band, new_band_name):
            """如果图像中缺少某个波段，则复制已存在的波段并重命名为新波段名"""
            return ee.Image(ee.Algorithms.If(
                band_names.contains(existing_band),
                image.addBands(image.select([existing_band]).rename(new_band_name)),
                image
            ))

        # 分别处理'VV'和'VH'波段
        s1_image = ee.Image(add_missing_band(s1_image, 'VV', 'VH'))
        s1_image = ee.Image(add_missing_band(s1_image, 'VH', 'VV'))

        # 选择'VV', 'VH', 和 'angle'波段
        return s1_image.select(['VV', 'VH', 'angle'])


# 重命名变量
def AssignVariablesKwargs(**kwargs):
    '''
    用于命名变量，给调试带来方便
    assign_variables_from_kwargs(DEM=DEMCOPERNICUS, Bands_=['VV_gamma0_flatDB', 'VH_gamma0_flatDB'])
    '''
    for var_name, value in kwargs.items():
        # 直接处理值，不需要使用eval，因为值已经是正确的Python数据类型
        globals()[var_name] = value
        
# 跨年份，选择数据
def filter_image_collection_by_dates(collection_name, date_ranges):
    """
    Filters an Earth Engine ImageCollection by multiple date ranges.

    Example usage: 
                    date_ranges = [
                        ('2020-06-01', '2020-09-30'),
                        ('2021-06-01', '2021-09-30'),
                        ('2023-06-01', '2023-09-30')]
                    filter_image_collection_by_dates('COPERNICUS/S2_SR', date_ranges)
    """

    collection = ee.ImageCollection(collection_name)
    filtered_collections = [collection.filterDate(start, end) for start, end in date_ranges]
    combined_collection = filtered_collections[0]
    for col in filtered_collections[1:]:
        combined_collection = combined_collection.merge(col)
    return combined_collection

# 从ee.ImageCollection中任意选取ee
def Select_imageNum(ImageCollection: ee.ImageCollection, i):
    return ee.Image(ImageCollection.toList(ImageCollection.size()).get(i))