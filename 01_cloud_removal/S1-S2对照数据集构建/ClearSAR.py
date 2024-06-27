# 抽取数据中的Clear以及SAR数据，形成文件夹和CSV文件
# 根据包含Clear、SAR的 CSV文件构建文件夹保存文件

import os
import pandas as pd
import shutil
from PackageDeepLearn.utils import DataIOTrans
from tqdm import trange


def Clear_SAR(CSV_path):
    ## 读取并提取信息
    DF = pd.read_csv(CSV_path)
    #  分别提取出正例和负例，然后通过文件夹序号拼接，SAR影像同样操作
    Positive = DF[DF['pos_neg'] == 1]

    ASCENDING = DF[DF['pos_neg'] == 2].drop(['parent', 'TimeStart_End', 'part'], axis=1)
    ASCENDING = ASCENDING.rename(columns={'tif': 'tif_ASCENDING', 'pos_neg': 'pos_neg_ASCENDING', 'date': 'date_ASCENDING'})

    DESCENDING = DF[DF['pos_neg'] == 3].drop(['parent', 'TimeStart_End', 'part'], axis=1)
    DESCENDING = DESCENDING.rename(
        columns={'tif': 'tif_DESCENDING', 'pos_neg': 'pos_neg_DESCENDING', 'date': 'date_DESCENDING'})


    def pd_merge(A=Positive, B=[ASCENDING, DESCENDING]):
        '''
        构建一个递归数据merge
        '''
        if len(B) == 0:
            return A
        A = pd.merge(A, B.pop(0), on=['num'], how='outer')
        return pd_merge(A, B)

    dataset = pd_merge()
    return dataset

def Copy_files(dataset,Input_Dir,Out_Dir):
    for i in trange(len(dataset)):
        ## 过滤，并生成路径
        Item = dataset.loc[i]
        # Origin
        Dir = os.path.join(Input_Dir,Item.TimeStart_End,Item.part,f'{int(Item.num):05d}')
        Clear = os.path.join(Dir,Item.tif)
        ASAR_path = os.path.join(Dir, Item.tif_ASCENDING)
        DSAR_path = os.path.join(Dir, Item.tif_DESCENDING)

        # Target
        Dir_out = os.path.join(Out_Dir,Item.TimeStart_End,Item.part,f'{int(Item.num):05d}')
        Clear_out = os.path.join(Dir_out,Item.tif)
        ASAR_path_out = os.path.join(Dir_out, Item.tif_ASCENDING)
        DSAR_path_out = os.path.join(Dir_out, Item.tif_DESCENDING)

        ## 创建文件夹，复制转移
        DataIOTrans.make_dir(os.path.dirname(Clear_out))
        shutil.copyfile(Clear,Clear_out)
        shutil.copyfile(ASAR_path, ASAR_path_out)
        shutil.copyfile(DSAR_path, DSAR_path_out)


if __name__=='__main__':

    CSV_path = r"H:\Datasets\2020-06-01_2020-09-30_processed\2020-06-01_2020-09-30.csv"
    Out_csvpath = r"H:\Datasets\2020-06-01_2020-09-30_processed\ClearSAR.csv"
    Input_Dir = r'H:\Datasets'
    Out_Dir = r'H:\Datasets\2020-06-01_2020-09-30_processed'

    # 生成ClearSAR.csv文件，检索合并了Clear\ASAR\DSAR
    dataset = Clear_SAR(CSV_path)

    # 过滤无效影像
    filt_dataset = dataset[(dataset['pos_neg']==1) & (dataset['pos_neg_ASCENDING']==2) & (dataset['pos_neg_DESCENDING']==3)
            & (dataset['tif'].notnull())]
    filt_dataset.to_csv(Out_csvpath)

    Copy_files(filt_dataset,Input_Dir,Out_Dir)


