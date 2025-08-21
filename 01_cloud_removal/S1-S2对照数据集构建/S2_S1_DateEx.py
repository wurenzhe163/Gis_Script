'''
    满足两种需求：
    1、根据已有的S2数据，导出S1数据需要的Date
    2、根据建成的S1-S2数据，构造新的csv文件，并构造数据集
'''

import pandas as pd
import warnings
import os,re
from PackageDeepLearn.utils import file_search_wash as fsw

warnings.filterwarnings('ignore')
#显示所有列
pd.set_option("display.max_columns",None)
#显示所有行
pd.set_option("display.max_rows",None)
#设置显示的最大列、宽等参数，消除打印不完全中间的省略号
pd.set_option("display.width",1000)


def create_dataframe(path,init=False):
    TiffPaths = fsw.search_files_alldir(path,endwith='.tif')
    TiffPaths_split = [each.split('\\') for each in TiffPaths]
    TiffPaths = [[os.path.join(*each[0:-4]),each[-4],each[-3],each[-2],each[-1]]for each in TiffPaths_split]
    tiffdf = pd.DataFrame(TiffPaths,columns=['parent','TimeStart_End','part','num','tif'])

    # 添加一列 正例和负例分别用1和0, ASCENDING_SAR用2表示 ,DESCENDING_SAR用3表示
    if init:
        tiffdf['pos_neg'] = [1 if 'Reprj_fillnodata_cloud' in each or 'Clear' in each else 0 for each in tiffdf.tif.to_numpy()]
    tiffdf['pos_neg'] = [1 if 'Reprj_fillnodata_cloud' in each or 'Clear' in each else 2
                         if 'ASCENDING' in each else 3
                         if 'DESCENDING' in each else 0
                         for each in tiffdf.tif.to_numpy()]
    return tiffdf

def caculate_date(tiffdf,csv_path=None):
    """
    计算成像日期
    """
    # 添加一列，成像日期
    tiffdf = tiffdf.reindex(columns = tiffdf.columns.tolist()+["date"])
    # 计算成像日期，其中cloud需要由同组其它数据计算
    for i,each in enumerate(tiffdf['pos_neg'].to_numpy()):

        if each == 1:
            if 'Clear' in tiffdf.tif[i]:
                # 对于真实图像，直接获取日期
                Date = re.search(r"\d{8}",tiffdf.tif[i]).group()
                tiffdf['date'][i] = pd.to_datetime(Date)
            elif 'Cloud' in tiffdf.tif[i]:
                # 对于合成图像，找出在同num中的其它图像的成像日期，获取平均成像日期
                datelist = pd.to_datetime([re.search(r"\d{8}",each).group()
                                           for each in tiffdf[tiffdf.num == tiffdf.num[i]][tiffdf.pos_neg==0].tif])
                tiffdf['date'][i] = datelist.mean()
            else:
                raise TypeError("正例既非‘Clear’也非‘Cloud’")

        if each == 0:
            Date = re.search(r"\d{8}", tiffdf.tif[i]).group()
            tiffdf['date'][i] = pd.to_datetime(Date)

        if each == 2 or each == 3:
            '''预留给SAR的接口'''
            Date = re.search(r"\d{8}", tiffdf.tif[i]).group()
            tiffdf['date'][i] = pd.to_datetime(Date)
    if csv_path:
        tiffdf.to_csv(csv_path, encoding='utf-8')
    else:
        return tiffdf


def create_dataset(data,output_path):
    '''
    data: PdDataFrame or csvPath
    '''
    if type(data) == str and data.split('.')[-1] == 'csv':
        DF = pd.read_csv(data)
    elif type(data) == pd.core.frame.DataFrame:
        DF = data
    else :
        print('Inpput data is wrong! Please check')

    DF = DF.drop(['Unnamed: 0'], axis=1)
    #  分别提取出正例和负例，然后通过文件夹序号拼接，SAR影像同样操作
    Positive = DF[DF['pos_neg'] == 1]

    Negtive = DF[DF['pos_neg'] == 0].drop([ 'part'], axis=1)
    Negtive = Negtive.rename(columns={'tif':'tif_Negtive','pos_neg':'pos_neg_Negtive','date':'date_Negtive'})

    ASCENDING = DF[DF['pos_neg'] == 2].drop([ 'part'], axis=1)
    ASCENDING = ASCENDING.rename(columns={'tif':'tif_ASCENDING','pos_neg':'pos_neg_ASCENDING','date':'date_ASCENDING'})

    DESCENDING = DF[DF['pos_neg'] == 3].drop(['part'], axis=1)
    DESCENDING = DESCENDING.rename(columns={'tif':'tif_DESCENDING','pos_neg':'pos_neg_DESCENDING','date':'date_DESCENDING'})

    def pd_merge(A=Positive,B=[Negtive,ASCENDING,DESCENDING]):
        '''
        构建一个递归数据merge
        '''
        if len(B)==0 :
            return A
        A = pd.merge(A, B.pop(0), on=['parent','TimeStart_End', 'num'], how='inner')
        return pd_merge(A,B)

    dataset = pd_merge()
    dataset.to_csv(output_path)


if __name__ == "__main__":

    path = r'H:\Cloud_restor_data'
    out_datacsv = r'H:\Cloud_restor_data\data.csv'
    out_datasetcsv = r'H:\Cloud_restor_data\data_concat.csv'
    S2_init = False   # 是否第一种场景(代码首行备注)

    # tiffdf = create_dataframe(path, init=S2_init)
    # caculate_date(tiffdf, csv_path=out_datacsv)

    if not S2_init:
        create_dataset(out_datacsv, out_datasetcsv)