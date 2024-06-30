"""
Detail:将GEE的S1和S2图像进行像素数量统计绘图，以及进行数据的标准化(提取均值、中位数、方差)，中位数根据统计图获取
Time:2023/2/10 15:54
Author:WRZ
"""
import os
from PackageDeepLearn.utils import file_search_wash,DataIOTrans, Visualize
import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def Boundary(df,ColumnName,y=1000):
    """
    将左右边界小于一定阈值的数据筛选出来,其中y是分母，代表了像素的缩放倍数
    此处y=1000，也就是寻找像素数小于1/1000的最大值和最小值
    """
    Boundaryvalue = int(sum(df[ColumnName]) / y)
    countFront = 0
    countBack = 0
    for counts, index in zip(df[ColumnName], df.index):
        if countFront + counts >= Boundaryvalue:
            indexFront = index  # 记录Index
            break
        else:
            countFront += counts

    for counts, index in zip(df[ColumnName].sort_index(ascending=False),
                             df.sort_index(ascending=False).index):
        if countBack + counts >= Boundaryvalue:
            indexBack = index  # 记录Index
            break
        else:
            countBack += counts

    return {'countFront':countFront,'indexFront':indexFront,
            'countBack':countBack,'indexBack':indexBack}

def cal_median(df,ColumnName):
    '''
    根据统计值计算中位数
    '''
    median_count = df[ColumnName].sum() / 2
    counts = 0
    for i,count in enumerate(df[ColumnName]):
        counts+=count
        if counts>= median_count:
            print('中位数计算：band{},索引为{},最终计数{},累计计数{}'.format(ColumnName,i,count,counts))
            return i
    print('check your inter')

def standardization(array, σ_, μ_):
    """
    Args:
        array: 矩阵 ndarray
        σ_: 均值 list
        μ_: 方差 list

    Returns:标准化后的矩阵
    """
    if len(array.shape) == 2:
        return (array - σ_) / μ_

    elif len(array.shape) == 3:
        array_ = np.zeros_like(array).astype(np.float64)
        h, w, c = array.shape
        for i in range(c):
            array_[:, :, i] = (array[:, :, i] - σ_[i]) / μ_[i]

    return array_

class DataNorm:
    """
    统计均值方差，以及统计每个像素值的个数，
    注意：DatawashPlot可以计算数据边界，联合NormImage_Clip函数可以进行归一化与标准化
    """

    def __init__(self, Csv_path,pos_neg='S2',Channels=None):
        # 需要操作的文件夹
        self.Csv = pd.read_csv(Csv_path)
        # 读取所有待求影像的路径
        if pos_neg == 'S2':
            self.Imgs = self.Csv[(self.Csv['pos_neg'] == 0) | (self.Csv['pos_neg'] == 1)]
        elif pos_neg == 'S1':
            self.Imgs = self.Csv[(self.Csv['pos_neg'] == 2) | (self.Csv['pos_neg'] == 3)]

        self.len_Imgs = len(self.Imgs)
        self.c = Channels

    def caculate_σ_μ_(self, ignoreValue=65535, axis=0):
        '''
        逐个计算每张图像的均值和方差，然后进行平均化
        计算数据集每个通道的均值和标准差，以及最大值最小值
        '''
        self.σ_ALL = [];
        self.μ_ALL = [];

        pbar = tqdm(self.Imgs.iterrows())
        for i, (_,each) in enumerate(pbar):
            Img_path = os.path.join(each.parent,each.TimeStart_End,each.part,f'{each.num:05d}',each.tif)
            array = DataIOTrans.DataIO.read_IMG(Img_path)
            array = array.astype(np.float32)
            array[array == ignoreValue] = np.nan

            if len(array.shape) == 2:
                '''
                转为shape==3
                '''
                array = array[..., np.newaxis]
                print('Your dataset shape=2')

            if i == 0:
                h, w, _ = array.shape
                if not self.c:
                    self.c = _
                array = array[...,0:self.c]
                array = array.reshape(h * w, self.c)
                self.arrayMax = array.max(axis=0)
                self.arrayMin = array.min(axis=0)
            else:
                h, w, _ = array.shape
                if not self.c:
                    self.c = _
                array = array[...,0:self.c]
                array = array.reshape(h * w, self.c)
                max = array.max(axis=axis)
                min = array.min(axis=axis)

                self.arrayMax[max > self.arrayMax] = max[max > self.arrayMax]
                self.arrayMin[min < self.arrayMin] = min[min < self.arrayMin]
            if ignoreValue:
                # # 这里出了bug大矩阵无法适用
                σ = np.array([np.nanmean(array[:, each]) for each in range(self.c)])
                μ = np.array([np.nanstd(array[:, each]) for each in range(self.c)])

            self.σ_ALL.append(σ)
            self.μ_ALL.append(μ)
            pbar.set_description(
                '轮次 :{}/{}.max:{},min:{},σ: {},μ:{}'.format(i + 1, self.len_Imgs, self.arrayMax, self.arrayMin, σ, μ))
        # # 统一向量长度
        # self.σ_ALL = [np.concatenate((each, [0., 0., 0., 0.]), axis=0) if len(each) == 12 else each for each in self.σ_ALL]
        # self.μ_ALL = [np.concatenate((each, [0., 0., 0., 0.]), axis=0) if len(each) == 12 else each for each in self.μ_ALL]

        self.σ = np.mean(np.array(self.σ_ALL), axis=0)
        self.μ = np.mean(np.array(self.μ_ALL), axis=0)
        return self.σ, self.μ

    # 上述的STD计算方式有一定错误，后期可以有空考虑补足，以下公式没有考虑空值的情况
    # def caculate_std(self,axis=0):
    #     mean = self.σ
    #     pbar = tqdm(self.Imgs)
    #     for i,each in enumerate(pbar):
    #         array = DataIOTrans.DataIO.read_IMG(each)
    #         if len(array.shape) == 2:
    #             array = array[..., np.newaxis]
    #             print('Your dataset shape=2')
    #         if i==0:
    #             h,w,c = array.shape
    #             MSE = np.prod((array-mean)**2/(h*w))
    #         else:
    #             MSE = MSE * np.prod((array-mean)**2/(h*w))
    #     return np.sqrt(MSE)

    def statisticValues(self):
        '''
        查看所有图像每个波段包含的数值以及数量
        '''
        pbar = tqdm(self.Imgs.iterrows())
        for i, (_,each) in enumerate(pbar):
            Img_path = os.path.join(each.parent, each.TimeStart_End, each.part, f'{each.num:05d}', each.tif)
            array = DataIOTrans.DataIO.read_IMG(Img_path)
            array = np.round(array, decimals=0)
            h, w, _ = array.shape
            if not self.c:
                self.c = _
            array = array[..., 0:self.c]
            if i == 0:
                ImgDict = dict([(str(each), 0) for each in range(self.c)])
            # 单波段
            for each_band in range(self.c):
                unique, count = np.unique(array[:, :, each_band], return_counts=True)
                data_count = dict(zip(unique.astype(str), count))
                if i == 0:
                    ImgDict[str(each_band)] = data_count
                else:
                    keys = ImgDict[str(each_band)].keys()
                    for key_, value_ in data_count.items():
                        if key_ in keys:
                            ImgDict[str(each_band)][key_] = ImgDict[str(each_band)][key_] + data_count[key_]
                        else:
                            ImgDict[str(each_band)].update({key_: value_})

            if i % 1000 == 0:
                print('len0={},len1={}'.format(len(ImgDict['0']), len(ImgDict['1'])))
        return ImgDict

    # def Denormalize(self,mean,std,max,min):
    #     '''
    #     将标准化的影像进行反标准化
    #     '''
    #     endarray = DataIOTrans.DataIO.read_IMG(each).astype(np.float64)
    #     newarray = endarray * (max - min) + min

class DatawashPlot:
    def __init__(self, picklePath):
        self.picklePath = picklePath
        with open(picklePath, 'rb') as f:
            self.Dictfile = pickle.load(f)
        print('ImageBandkeys={}'.format(self.Dictfile.keys()))
        # 合并同类项(0.0，-0.0此类情况，去除Nan)
        for eachkey in self.Dictfile.keys():
            try:
                print('key={} 0.0 values is combine -> {} + {}'.format(eachkey, self.Dictfile[eachkey]['0.0'],
                                                                       self.Dictfile[eachkey]['-0.0']))
                self.Dictfile[eachkey]['0.0'] += self.Dictfile[eachkey]['-0.0']
                del self.Dictfile[eachkey]['-0.0']
            except:
                pass
            try:
                print('key={} delete nan length = {}'.format(eachkey, self.Dictfile[eachkey]['nan']))
                del self.Dictfile[eachkey]['nan']
            except:
                pass
        # 字符转为数值，并排序，建立Dataframe
        self.df = pd.DataFrame(self.Dictfile)
        self.df.index = self.df.index.astype('float64')
        self.df = self.df.sort_index(axis=0, ascending=True)
        self.df['values'] = self.df.index
        self.df.to_csv(os.path.basename(picklePath) + '.csv',index=False)

    def histPlot(self, picklePath='S2_filter.pickle',csvpath=None):
        """
        分布直方图绘制,约定bar数量为60个(未做)
        """
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 300  # 分辨率
        plt.figure(figsize=(5, 5))
        saveDict = {}
        if csvpath:
            self.df = pd.read_csv(csvpath)
        for eachColumns in self.df.columns:
            if eachColumns != 'values':
                # 创建新的dataframe用于单独过滤Nan值
                newdf = pd.DataFrame(self.df, columns=[eachColumns, 'values']).fillna(0)#.dropna()
                newdf.index = newdf.index.astype('float64')
                newdf = newdf.sort_index(ascending=True)
                ax = sns.barplot(x='values', y=eachColumns, data=newdf)  # ,data=newdf
                dflens = len(newdf)

                # 60-120 等分合并，数据量大于等于120，则两两合并，最多缩减至60

                # # 10 等分标注
                s = [each for each in range(dflens) if each % (dflens // 10) == 0]
                plt.xticks(ticks=s, labels=newdf['values'][s], rotation=45)

                # 计算需要过滤的数值，画红线,前0.1%后0.1%，注意画图的时候需要前移后移一个数值
                Dfbound = Boundary(newdf, eachColumns)
                ax.axvline(x=np.where(newdf.index == Dfbound['indexFront'])[0][0] + 0.5, c='red', ls='--', lw=2)
                ax.axvline(x=np.where(newdf.index == Dfbound['indexBack'])[0][0] - 0.5, c='red', ls='--', lw=2)

                # plt.show()
                # 保存
                saveDict[eachColumns] = Dfbound
                plt.savefig(eachColumns + '.jpg', dpi=300, bbox_inches='tight')
                plt.clf()

        # 保存过滤后的pickle文件
        with open(os.path.dirname(self.picklePath) + '/' + picklePath, 'wb') as f:
            pickle.dump(saveDict, f)

def NormImage_Clip(CountDFPath,ClippicklePath):

    Counts = pd.read_csv(CountDFPath).fillna(0)

    with open(ClippicklePath, 'rb') as f:
        Clip = pickle.load(f)

    # Clip数据
    for eachColumns in Clip.keys():
        indexFront = Clip[eachColumns]['indexFront']
        indexBack = Clip[eachColumns]['indexBack']
        # 将超限的数值个数累加到最靠近界限的阈值上，并将超限值的个数设为0
        for i in range(len(Counts)):
            if i < indexFront:
                Counts.loc[indexFront, eachColumns] += Counts.loc[i, eachColumns]
                Counts.loc[i, eachColumns] = 0
            if i > indexBack:
                Counts.loc[indexBack, eachColumns] += Counts.loc[i, eachColumns]
                Counts.loc[i, eachColumns] = 0
        # 丢弃行
        # Counts.drop([i for i in range(len(Counts)) if i>indexBack or i < indexFront])
    # 归一化之后，统计均值、中位数、方差，以列表方式返回
    mean = []; median = []; std = [];stdmedian=[]; min=[]; max=[]
    for eachColumns in Clip.keys():
        indexFront = Clip[eachColumns]['indexFront']
        indexBack = Clip[eachColumns]['indexBack']
        valuesFront = Counts['values'][indexFront]
        valuesBack  = Counts['values'][indexBack]

        # 使用界限值进行归一化
        values = (Counts['values'] - valuesFront) / (valuesBack - valuesFront) #所有像素值之和

        meanValue = ((Counts[eachColumns] * values).sum() / Counts[eachColumns].sum())
        medianValue = values[cal_median(Counts, eachColumns)]
        stdValue = np.sqrt((Counts[eachColumns] * (values-meanValue)*(values-meanValue)).sum()/ Counts[eachColumns].sum())
        stdmedianValue = np.sqrt((Counts[eachColumns] * (values-medianValue)*(values-medianValue)).sum()/ Counts[eachColumns].sum())

        mean.append(meanValue)
        median.append(medianValue)
        std.append(stdValue)
        stdmedian.append(stdmedianValue)
        min.append(valuesFront)
        max.append(valuesBack)
    return Counts,mean,median,std,stdmedian,min,max

if __name__ == '__main__':
    # # 统计全部图像(除Nodata)最大值，以及最小值；统计所有图像的均值和方差，并计算平均图像均值和方差
    S2 = DataNorm(Csv_path=r'E:\Datasets\2020-06-01_2020-09-30_processed\2020-06-01_2020-09-30.csv',
                    pos_neg='S2',
                    Channels=12)
    # S2.caculate_σ_μ_()
    # SaveDict = {'σ_ALL': S2.σ_ALL, 'μ_ALL': S2.μ_ALL, 'σ': S2.σ, 'μ': S2.μ, 'arrayMax': S2.arrayMax,
    #             'arrayMin': S2.arrayMin}
    # with open(r'E:\Datasets\2020-06-01_2020-09-30_processed\Alpha_mean_max_min.pickle', 'wb') as f:
    #     pickle.dump(SaveDict, f)

    # 统计所有像素值的个数
    ImageDict=S2.statisticValues()
    with open(r'E:\Datasets\2020-06-01_2020-09-30_processed\S2.pickle', 'wb') as f:
        pickle.dump(ImageDict, f)

    # 根据统计的像素个数(pickle)生成csv，并过滤生成filter, filter包含了像素数量的前1/1000,以及像素数量的后1/1000
    '''
    {'countFront': 0,
     'indexFront': 0.0,
     'countBack': 14035437.0,
     'indexBack': 14298.0}
    这些数据属于极少出现的波动数据，会给影像运算带来误差，故舍弃
    注意:由于代码不完善，需要手动删除第一列第一行csv数据
    '''
    Data = DatawashPlot(r'E:\Datasets\2020-06-01_2020-09-30_processed\S2.pickle')
    Data.histPlot(picklePath = r'S2_filter.pickle',
                  csvpath=r'E:\Datasets\2020-06-01_2020-09-30_processed\S2.pickle.csv')

    # 数据阈值裁剪、归一化，并计算mean,median,std,stdmedian
    CountDFPath = r'E:\Datasets\2020-06-01_2020-09-30_processed\S2\S2.pickle.csv'
    ClippicklePath = r'E:\Datasets\2020-06-01_2020-09-30_processed\S2\S2_filter.pickle'
    CountsDF,mean,median,std,stdmedian,min,max = NormImage_Clip(CountDFPath, ClippicklePath)
    S2_Statistic = {'S2_valuesFront':min,
     'S2_valuesBack':max,
     'S2_mean':mean,'S2_median':median,
     'S2_std':std,'S2_stdmedian':stdmedian}
    pd.DataFrame(S2_Statistic).to_csv(r'E:\Datasets\2020-06-01_2020-09-30_processed\S2_Statistic.csv',index=False)






    #--------------------------------------------------S1----------------------------------------------#
    # 统计全部图像(除Nodata)最大值，以及最小值；统计所有图像的均值和方差，并计算平均图像均值和方差
    S1 = DataNorm(Csv_path=r'E:\Datasets\2020-06-01_2020-09-30_processed\2020-06-01_2020-09-30.csv',
                    pos_neg='S1',
                    Channels=3)
    S1.caculate_σ_μ_()
    SaveDict = {'σ_ALL': S1.σ_ALL, 'μ_ALL': S1.μ_ALL, 'σ': S1.σ, 'μ': S1.μ, 'arrayMax': S1.arrayMax,
                'arrayMin': S1.arrayMin}
    with open(r'E:\Datasets\2020-06-01_2020-09-30_processed\Alpha_mean_max_min.pickle', 'wb') as f:
        pickle.dump(SaveDict, f)

    # 统计所有像素值的个数
    ImageDict=S1.statisticValues()
    with open(r'E:\Datasets\2020-06-01_2020-09-30_processed\S1.pickle', 'wb') as f:
        pickle.dump(ImageDict, f)

    # 根据统计的像素个数(pickle)生成csv，并过滤生成filter, filter包含了像素数量的前1/1000,以及像素数量的后1/1000
    '''
    {'countFront': 0,
     'indexFront': 0.0,
     'countBack': 14035437.0,
     'indexBack': 14298.0}
    这些数据属于极少出现的波动数据，会给影像运算带来误差，故舍弃
    注意:由于代码不完善，需要手动删除第一列第一行csv数据
    '''
    Data = DatawashPlot(r'E:\Datasets\2020-06-01_2020-09-30_processed\S1.pickle')
    Data.histPlot(picklePath = r'S1_filter.pickle',
                  csvpath=r'E:\Datasets\2020-06-01_2020-09-30_processed\S1.pickle.csv')

    # 数据阈值裁剪、归一化，并计算mean,median,std,stdmedian
    CountDFPath = r'E:\Datasets\2020-06-01_2020-09-30_processed\S1\S1.pickle.csv'
    ClippicklePath = r'E:\Datasets\2020-06-01_2020-09-30_processed\S1\S1_filter.pickle'
    CountsDF,mean,median,std,stdmedian,min,max = NormImage_Clip(CountDFPath, ClippicklePath)
    S1_Statistic = {'S1_valuesFront':min,
     'S1_valuesBack':max,
     'S1_mean':mean,'S1_median':median,
     'S1_std':std,'S1_stdmedian':stdmedian}
    pd.DataFrame(S1_Statistic).to_csv(r'E:\Datasets\2020-06-01_2020-09-30_processed\S1_Statistic.csv',index=False)
