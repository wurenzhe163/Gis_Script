import numpy as np
import os
from osgeo import gdal

search_files = lambda path : sorted([os.path.join(path,f) for f in os.listdir(path)])

def make_dir(path):
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return path
    return path

class Denormalize(object):
    '''
    return : 反标准化
    '''
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return transforms.functional.normalize(tensor, self._mean, self._std)

class DataTrans(object):
    @staticmethod
    def OneHotEncode(LabelImage,NumClass):
        '''
        Onehot Encoder  2021/03/23 by Mr.w
        -------------------------------------------------------------
        LabelImage ： Ndarry   |   NumClass ： Int
        -------------------------------------------------------------
        return： Ndarry
        '''
        one_hot_codes = np.eye(NumClass).astype(np.uint8)
        try:
            one_hot_label = one_hot_codes[LabelImage]
        except IndexError:
            # pre treat cast 不连续值 到 连续值
            Unique_code = np.unique(LabelImage)
            Objectives_code = np.arange(len(Unique_code))
            for i, j in zip(Unique_code, Objectives_code):
                LabelImage[LabelImage == i] = j
                # print('影像编码从{},转换为{}'.format(i, j))

            one_hot_label = one_hot_codes[LabelImage]
        # except IndexError:
        #     # print('标签存在不连续值，最大值为{}--->已默认将该值进行连续化,仅适用于二分类'.format(np.max(LabelImage)))
        #     LabelImage[LabelImage == np.max(LabelImage)] = 1
        #     one_hot_label = one_hot_codes[LabelImage]
        return one_hot_label

    @staticmethod
    def OneHotDecode(OneHotImage):
        '''
        OneHotDecode 2021/03/23 by Mr.w
        -------------------------------------------------------------
        OneHotImage : ndarray -->(512,512,x)
        -------------------------------------------------------------
        return : image --> (512,512)
        '''
        return np.argmax(OneHotImage,axis=-1)

    @staticmethod
    def StandardScaler(array, mean, std):
        """
        Z-Norm
        Args:
            array: 矩阵 ndarray
            mean: 均值 list
            std: 方差 list
        Returns:标准化后的矩阵
        """
        if len(array.shape) == 2:
            return (array - mean) / std

        elif len(array.shape) == 3:
            array_ = np.zeros_like(array).astype(np.float64)
            h, w, c = array.shape
            for i in range(c):
                array_[:, :, i] = (array[:, :, i] - mean[i]) / std[i]
        return array_

    @staticmethod
    def MinMaxScaler(array, max, min):
        '''归一化'''
        if len(array.shape) == 2:
            return (array - min) / (max-min)

        elif len(array.shape) == 3:
            array_ = np.zeros_like(array).astype(np.float64)
            h, w, c = array.shape
            for i in range(c):
                array_[:, :, i] = (array[:, :, i] - min[i]) / (max[i]-min[i])
        return array_

    @staticmethod
    def MinMax_Standard(array, max:list, min:list,mean:list,std:list):
        '''
        先归一化，再标准化.
        注意该标准化所用的mean和std均是归一化之后计算值
        '''
        if len(array.shape) == 2:
            _ = (array - min) / (max - min)
            return (_ - mean) / std

        elif len(array.shape) == 3:
            array_1 = np.zeros_like(array).astype(np.float64)
            array_2 = np.zeros_like(array).astype(np.float64)
            h, w, c = array.shape
            for i in range(c):
                array_1[:, :, i] = (array[:, :, i] - min[i]) / (max[i]-min[i])
                array_2[:, :, i] = (array_1[:, :, i] - mean[i]) / std[i]
        return array_2

    @staticmethod
    def MinMaxArray(array):
        '''计算最大最小值'''

        if len(array.shape) == 2:
            array = array[...,None]

        if len(array.shape) == 3:
            h, w, c = array.shape
            max = [];min=[]
            for i in range(c):
                max.append(np.max(array[:,:,i]))
                min.append(np.min(array[:,:,i]))
        else:
            print('array.shape is wrong')

        return max,min

    @staticmethod
    def copy_geoCoordSys(img_pos_path, img_none_path):
        '''
        获取img_pos坐标，并赋值给img_none
        :param img_pos_path: 带有坐标的图像
        :param img_none_path: 不带坐标的图像
        '''

        def def_geoCoordSys(read_path, img_transf, img_proj):
            array_dataset = gdal.Open(read_path)
            img_array = array_dataset.ReadAsArray(0, 0, array_dataset.RasterXSize, array_dataset.RasterYSize)
            if 'int8' in img_array.dtype.name:
                datatype = gdal.GDT_Byte
            elif 'int16' in img_array.dtype.name:
                datatype = gdal.GDT_UInt16
            else:
                datatype = gdal.GDT_Float32

            if len(img_array.shape) == 3:
                img_bands, im_height, im_width = img_array.shape
            else:
                img_bands, (im_height, im_width) = 1, img_array.shape

            filename = read_path[:-4] + '_proj' + read_path[-4:]
            driver = gdal.GetDriverByName("GTiff")  # 创建文件驱动
            dataset = driver.Create(filename, im_width, im_height, img_bands, datatype)
            dataset.SetGeoTransform(img_transf)  # 写入仿射变换参数
            dataset.SetProjection(img_proj)  # 写入投影

            # 写入影像数据
            if img_bands == 1:
                dataset.GetRasterBand(1).WriteArray(img_array)
            else:
                for i in range(img_bands):
                    dataset.GetRasterBand(i + 1).WriteArray(img_array[i])
            print(read_path, 'geoCoordSys get!')

        dataset = gdal.Open(img_pos_path)  # 打开文件
        img_pos_transf = dataset.GetGeoTransform()  # 仿射矩阵
        img_pos_proj = dataset.GetProjection()  # 地图投影信息
        def_geoCoordSys(img_none_path, img_pos_transf, img_pos_proj)

class DataIO(object):
    @staticmethod
    def _get_dir(*path,DATA_DIR=r''):
        """
        as : a = ['train', 'val', 'test'] ; _get_dir(*a,DATA_DIR = 'D:\\deep_road\\tiff')
        :return list path
        """
        return [os.path.join(DATA_DIR, each) for each in path]

    @staticmethod
    def read_IMG(path,flag=0):
        """
        读为一个numpy数组,读取所有波段
        对于RGB图像仍然是RGB通道，cv2.imread读取的是BGR通道
        path : img_path as:c:/xx/xx.tif
        flag:0不反回坐标，1返回坐标
        """
        dataset = gdal.Open(path)
        if dataset == None:
            raise Exception("Unable to read the data file")

        nXSize = dataset.RasterXSize  # 列数
        nYSize = dataset.RasterYSize  # 行数
        bands = dataset.RasterCount  # 波段
        Raster1 = dataset.GetRasterBand(1)
        if flag==1:
            img_transf = dataset.GetGeoTransform()  # 仿射矩阵
            img_proj = dataset.GetProjection()  # 地图投影信息

        if Raster1.DataType == 1 :
            datatype = np.uint8
        elif Raster1.DataType == 2:
            datatype = np.uint16
        else:
            datatype = float

        data = np.zeros([nYSize, nXSize, bands], dtype=datatype)
        for i in range(bands):
            band = dataset.GetRasterBand(i + 1)
            data[:, :, i] = band.ReadAsArray(0, 0, nXSize, nYSize)  # .astype(np.complex)
        if flag==0:
            return data
        elif flag==1:
            return data,img_transf,img_proj
        else:
            print('None Output, please check')



    @staticmethod
    def save_Gdal(img_array, SavePath, transf=False, img_transf=None, img_proj=None):
        """

        Args:
            img_array:  [H,W,C] , RGB色彩，不限通道深度
            SavePath:
            transf: 是否进行投影
            img_transf: 仿射变换
            img_proj: 投影信息

        Returns: 0

        """
        dirname = os.path.dirname(SavePath)
        if os.path.isabs(dirname):
            make_dir(dirname)

        # 判断数据类型
        if 'int8' in img_array.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in img_array.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判断数据维度，仅接受shape=3或shape=2
        if len(img_array.shape) == 3:
            im_height, im_width, img_bands = img_array.shape
        else:
            img_bands, (im_height, im_width) = 1, img_array.shape

        driver = gdal.GetDriverByName("GTiff")  # 创建文件驱动
        dataset = driver.Create(SavePath, im_width, im_height, img_bands, datatype)

        if transf:
            dataset.SetGeoTransform(img_transf)  # 写入仿射变换参数
            dataset.SetProjection(img_proj)  # 写入投影

        # 写入影像数据
        if len(img_array.shape) == 2:
            dataset.GetRasterBand(1).WriteArray(img_array)
        else:
            for i in range(img_bands):
                dataset.GetRasterBand(i + 1).WriteArray(img_array[:, :, i])

        dataset = None
