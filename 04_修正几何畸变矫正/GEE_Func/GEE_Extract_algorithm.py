import ee
from functools import partial
from skimage.filters import threshold_minimum
from scipy import ndimage as ndi
import numpy as np
import geemap
import sys



class Polarization_comb(object):
    '''极化组合，暂时没有想好怎么用'''
    pass

class Cluster_extract(object):
    '''聚类提取'''
    @staticmethod
    def afn_Kmeans(inputImg, defaultStudyArea, numberOfUnsupervisedClusters=2, nativeScaleOfImage=10, numPixels=1000):
        '''
        inputImg：任何图像，所有波段将用于聚类。
        numberOfUnsupervisedClusters：要创建的聚类数的可调参数。
        defaultStudyArea：训练样本的默认区域。
        nativeScaleOfImage：图像的本地比例尺。
        '''
        # Make a new sample set on the inputImg. Here the sample set is randomly selected spatially.并训练
        training = inputImg.sample(region=defaultStudyArea, scale=nativeScaleOfImage, numPixels=numPixels)
        cluster = ee.Clusterer.wekaKMeans(numberOfUnsupervisedClusters).train(training)
        # Now apply that clusterer to the raw image that was also passed in.
        toexport = inputImg.cluster(cluster)
        # 选择分类结果，并进行重命名
        clusterUnsup = toexport.select(0).rename('unsupervisedClass')
        return clusterUnsup

    @staticmethod
    def afn_Cobweb(inputImg, defaultStudyArea, cutoff=0.004, nativeScaleOfImage=10, numPixels=1000):
        # 这个算法效果一般
        training = inputImg.sample(region=defaultStudyArea, scale=nativeScaleOfImage, numPixels=numPixels)
        cluster = ee.Clusterer.wekaCobweb(cutoff=cutoff).train(training)
        toexport = inputImg.cluster(cluster)
        # 选择分类结果，并进行重命名
        clusterUnsup = toexport.select(0).rename('unsupervisedClass')
        return clusterUnsup

    @staticmethod
    def afn_Xmeans(inputImg, defaultStudyArea, numberOfUnsupervisedClusters=2, nativeScaleOfImage=10, numPixels=1000):
        training = inputImg.sample(region=defaultStudyArea, scale=nativeScaleOfImage, numPixels=numPixels)
        cluster = ee.Clusterer.wekaXMeans(maxClusters=numberOfUnsupervisedClusters).train(training)
        toexport = inputImg.cluster(cluster)
        clusterUnsup = toexport.select(0).rename('unsupervisedClass')
        return clusterUnsup

    @staticmethod
    def afn_LVQ(inputImg, defaultStudyArea, numberOfUnsupervisedClusters=2, nativeScaleOfImage=10, numPixels=1000):
        training = inputImg.sample(region=defaultStudyArea, scale=nativeScaleOfImage, numPixels=numPixels)
        cluster = ee.Clusterer.wekaLVQ(numClusters=numberOfUnsupervisedClusters).train(training)
        toexport = inputImg.cluster(cluster)
        clusterUnsup = toexport.select(0).rename('unsupervisedClass')
        return clusterUnsup

    @staticmethod
    def afn_CascadeKMeans(inputImg, defaultStudyArea, numberOfUnsupervisedClusters=2, nativeScaleOfImage=10,
                          numPixels=1000):
        training = inputImg.sample(region=defaultStudyArea, scale=nativeScaleOfImage, numPixels=numPixels)
        cluster = ee.Clusterer.wekaCascadeKMeans(maxClusters=numberOfUnsupervisedClusters).train(training)
        toexport = inputImg.cluster(cluster)
        clusterUnsup = toexport.select(0).rename('unsupervisedClass')
        return clusterUnsup

    @staticmethod
    def afn_SNIC(imageOriginal, SuperPixelSize=10, Compactness=1, Connectivity=4, NeighborhoodSize=20,
                 SeedShape='square'):
        '''
        下面是对 afn_SNIC 函数的参数进行解释：
        imageOriginal: 要进行超像素分割的原始图像。这是必需的参数。
        SuperPixelSize: default=5意思是超像素大小，即生成的超像素的平均大小。通常情况下，该值取决于你要处理的图像的大小和分辨率。如果原始图像比较大，那么超像素的大小可以设置得较大，例如100或200像素。如果原始图像比较小，那么超像素的大小应设置得比较小，例如10或20像素。
        Compactness: default=1该参数控制着每个超像素的形态。它的值通常在[1,20]范围内取值。Compactness值越小，超像素类别的形状越规则，而 Compactness值越大，超像素的形状越灵活。
        Connectivity: default=8连通性参数，描述每个像素与周围像素的关系，这个参数通常设为8。
        NeighborhoodSize：瓦片邻域大小（为了避免瓦片边界伪影）。默认为2 * size。定义像素颜色在计算距离时考虑的领域大小。值越大，超像素越平滑，边缘越模糊；值越小，超像素分割边缘更为锐利。
        SeedShape：种子形状，可以是“square”或“hex”。
        '''
        theSeeds = ee.Algorithms.Image.Segmentation.seedGrid(SuperPixelSize, SeedShape)

        snic2 = ee.Algorithms.Image.Segmentation.SNIC(image=imageOriginal,
                                                      size=SuperPixelSize,
                                                      compactness=Compactness,
                                                      connectivity=Connectivity,
                                                      neighborhoodSize=NeighborhoodSize,
                                                      seeds=theSeeds)
        theStack = snic2.addBands(theSeeds)
        return theStack

class Adaptive_threshold(object):
    @staticmethod
    def afn_otsu(histogram):
        '''大津法，类间方差最大化'''
        # 各组频数
        counts = ee.Array(ee.Dictionary(histogram).get('histogram'))
        # 各组的值
        means = ee.Array(ee.Dictionary(histogram).get('bucketMeans'))
        # 组数
        size = means.length().get([0])
        # 总像元数量
        total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
        # 所有组的值之和
        sum_ = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
        # 整幅影像的均值
        mean_ = sum_.divide(total)
        # 与组数相同长度的索引
        indices = ee.List.sequence(1, size)

        def calc_bss(i, sum_, mean_):
            '''穷举法计算类内方差'''
            aCounts = counts.slice(0, ee.Number(i))
            # 从i分割为两类A、B 计算A方差
            aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0])
            aMeans = means.slice(0, ee.Number(i))
            # 类别A均值
            aMean = aMeans.multiply(aCounts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(aCount)
            bCount = total.subtract(aCount)
            # 类别B均值
            bMean = sum_.subtract(aCount.multiply(aMean)).divide(bCount)
            # 类间方差
            return aCount.multiply(aMean.subtract(mean_).pow(2)).add(bCount.multiply(bMean.subtract(mean_).pow(2)))

        # 计算类内方差
        bss = indices.map(partial(calc_bss, sum_=sum_, mean_=mean_))
        # 排序选出最适阈值

        return means.sort(bss).get([-1])

    @staticmethod
    def afn_histPeak(img, region=None,default_value=1):
        '''
        采用skimage计算双峰
        参考论文：An analysis of histogram-based thresholding algorithms
        The analysis of cell images
        '''
        img_numpy = geemap.ee_to_numpy(img, region=region,default_value=default_value)  # region必须是矩形
        threshold = threshold_minimum(img_numpy)
        return threshold

    @staticmethod
    def my_threshold_minimum(bin_centers, counts,max_num_iter = 10000):
        '''根据直方图运算，可以忽视空值'''
        def find_local_maxima_idx(hist):
            maximum_idxs = list()
            direction = 1
            for i in range(hist.shape[0] - 1):
                if direction > 0:
                    if hist[i + 1] < hist[i]:
                        direction = -1
                        maximum_idxs.append(i)
                else:
                    if hist[i + 1] > hist[i]:
                        direction = 1
            return maximum_idxs

        smooth_hist = counts.astype('float32', copy=False)
        for counter in range(max_num_iter ):
            smooth_hist = ndi.uniform_filter1d(smooth_hist, 3)
            maximum_idxs = find_local_maxima_idx(smooth_hist)
            if len(maximum_idxs) < 3:
                break

        if len(maximum_idxs) != 2:
            raise RuntimeError('Unable to find two maxima in histogram')
        elif counter == max_num_iter - 1:
            raise RuntimeError('Maximum iteration reached for histogram' 'smoothing')

        # Find lowest point between the maxima
        threshold_idx = np.argmin(smooth_hist[maximum_idxs[0]:maximum_idxs[1] + 1])
        return bin_centers[maximum_idxs[0] + threshold_idx]

    @staticmethod
    def my_threshold_yen(bin_centers, counts):
        '''根据直方图运算，可以忽视空值'''
        # Calculate probability mass function
        pmf = counts.astype('float32', copy=False) / counts.sum()
        P1 = np.cumsum(pmf)  # Cumulative normalized histogram
        P1_sq = np.cumsum(pmf ** 2)
        # Get cumsum calculated from end of squared array:
        P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
        # P2_sq indexes is shifted +1. I assume, with P1[:-1] it's help avoid
        # '-inf' in crit. ImageJ Yen implementation replaces those values by zero.
        crit = np.log(((P1_sq[:-1] * P2_sq[1:]) ** -1) *
                      (P1[:-1] * (1.0 - P1[:-1])) ** 2)
        return bin_centers[crit.argmax()]

    @staticmethod
    def my_threshold_isodata(bin_centers, counts, bin_width , returnAll=False):
        counts = counts.astype('float32', copy=False)
        csuml = np.cumsum(counts)
        csumh = csuml[-1] - csuml

        # intensity_sum contains the total pixel intensity from each bin
        intensity_sum = counts * bin_centers
        csum_intensity = np.cumsum(intensity_sum)
        lower = csum_intensity[:-1] / csuml[:-1]
        higher = (csum_intensity[-1] - csum_intensity[:-1]) / (csumh[:-1]+sys.float_info.min)
        all_mean = (lower + higher) / 2.0
        distances = all_mean - bin_centers[:-1]
        thresholds = bin_centers[:-1][(distances >= 0) & (distances < bin_width)]
        if len(thresholds) == 0:
            thresholds = [bin_centers[:-1][distances >= 0][-1]]
        if returnAll:
            return thresholds
        else:
            return thresholds[0]

class Supervis_classify(object):
    pass

class Reprocess(object):
    '''图像后处理算法'''
    @staticmethod
    def Open_close(img, radius=10):
        uniformKernel = ee.Kernel.square(**{'radius': radius, 'units': 'meters'})
        min = img.reduceNeighborhood(**{'reducer': ee.Reducer.min(), 'kernel': uniformKernel})
        Openning = min.reduceNeighborhood(**{'reducer': ee.Reducer.max(), 'kernel': uniformKernel})
        max = Openning.reduceNeighborhood(**{'reducer': ee.Reducer.max(), 'kernel': uniformKernel})
        Closing = max.reduceNeighborhood(**{'reducer': ee.Reducer.min(), 'kernel': uniformKernel})
        return Closing

    @staticmethod
    def image2vector(result, resultband=0, radius=10, GLarea=1., scale=10,FilterBound=None, del_maxcount=False):

        # 图像学运算，避免噪点过多，矢量化失败
        Closing_result = Reprocess.Open_close(result.select(resultband), radius = radius)
        # 分类图转为矢量并删除背景，添加select(0)会减少bug，不晓得为啥
        if GLarea > 20:
            Vectors = Closing_result.select(0).reduceToVectors(scale=scale*3, geometryType='polygon',
                                                               eightConnected=True)
        else:
            Vectors = Closing_result.select(0).reduceToVectors(scale=scale, geometryType='polygon',
                                                               eightConnected=True)
        if del_maxcount:
            Max_count = Vectors.aggregate_max('count')
            Vectors = Vectors.filterMetadata('count', 'not_equals', Max_count)
        # 提取分类结果,并合并为一个矢量
        Extract = Vectors.filterBounds(FilterBound)
        Union_ex = ee.Feature(Extract.union(1).first())

        return Union_ex

