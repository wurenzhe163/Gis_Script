from qgis.core import (QgsProcessingAlgorithm,
                       QgsProcessingParameterFolderDestination,
                       QgsProcessingParameterFile,
                       QgsProcessingParameterBoolean,
                       QgsProcessingParameterString,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterRasterDestination,
                       QgsProcessingException,
                       QgsApplication)
from qgis import processing
import os
from osgeo import gdal

class MergeTiffAlgorithm(QgsProcessingAlgorithm):
    INPUT_FOLDER = 'INPUT_FOLDER'
    TO_EPSG_4326 = 'TO_EPSG_4326'
    OUTPUT_TIF = 'OUTPUT_TIF'
    KEEP_VRT = 'KEEP_VRT'
    OUTPUT_TYPE = 'OUTPUT_TYPE'
    COMPRESSION = 'COMPRESSION'

    OUTPUT_TYPES = [
        'Byte', 'UInt16', 'Int16', 'UInt32', 'Int32', 
        'Float32', 'Float64', 'CInt16', 'CInt32', 'CFloat32', 'CFloat64'
    ]

    COMPRESSION_OPTIONS = ['None', 'LZW', 'DEFLATE', 'PACKBITS']

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_FOLDER,
                'Input Folder',
                behavior=QgsProcessingParameterFile.Folder
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.TO_EPSG_4326,
                'Transform image projection to EPSG:4326',
                defaultValue=False
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_TIF,
                'Output Merged TIFF File'
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.KEEP_VRT,
                'Keep VRT File',
                defaultValue=False
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.OUTPUT_TYPE,
                'Output Data Type',
                options=self.OUTPUT_TYPES,
                defaultValue=0
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.COMPRESSION,
                'TIFF Compression',
                options=self.COMPRESSION_OPTIONS,
                defaultValue=1  # Default to LZW
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        input_folder = self.parameterAsFile(
            parameters,
            self.INPUT_FOLDER,
            context
        )

        to_epsg_4326 = self.parameterAsBoolean(
            parameters,
            self.TO_EPSG_4326,
            context
        )

        output_tif = self.parameterAsOutputLayer(
            parameters,
            self.OUTPUT_TIF,
            context
        )

        keep_vrt = self.parameterAsBoolean(
            parameters,
            self.KEEP_VRT,
            context
        )

        output_type = self.OUTPUT_TYPES[self.parameterAsEnum(
            parameters,
            self.OUTPUT_TYPE,
            context
        )]

        compression = self.COMPRESSION_OPTIONS[self.parameterAsEnum(
            parameters,
            self.COMPRESSION,
            context
        )]

        def search_files(path, endwith='.tif'):
            return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(endwith)]

        def check_projection(input_group):
            projections = []
            for file_path in input_group:
                dataset = gdal.Open(file_path)
                projections.append(dataset.GetProjection())
                dataset = None
            return projections

        os.chdir(input_folder)
        input_group = search_files(path=input_folder, endwith='.tif')

        if not input_group:
            raise QgsProcessingException('No TIFF files found in the specified folder.')

        projections = check_projection(input_group)

        output_vrt = os.path.join(input_folder, 'temp.vrt')

        if to_epsg_4326:
            aligned_inputs = []
            for img in input_group:
                aligned_img = img.replace('.tif', '_aligned.tif')
                gdal.Warp(aligned_img, img, dstSRS='EPSG:4326', outputType=gdal.GetDataTypeByName(output_type))  # 转换至 WGS84 投影
                aligned_inputs.append(aligned_img)
        else:
            aligned_inputs = input_group

        # 创建虚拟数据集
        gdal.BuildVRT(output_vrt, aligned_inputs)

        # 使用期望的重采样算法进行图像拼接
        gdal.Warp(output_tif, output_vrt, resampleAlg='bilinear', outputType=gdal.GetDataTypeByName(output_type), options=['COMPRESS=' + compression]) #'average', 'nearest', 'bilinear', 'cubic', 'cubicspline', 'lanczos', 'average_magphase', 'mode'

        if not keep_vrt:
            os.remove(output_vrt)
        if to_epsg_4326:
            for img in aligned_inputs:
                os.remove(img)

        return {self.OUTPUT_TIF: output_tif}

    def name(self):
        return 'mergetiff'

    def displayName(self):
        return 'Merge TIFF Images'

    def group(self):
        return 'Raster'

    def groupId(self):
        return 'raster'

    def createInstance(self):
        return MergeTiffAlgorithm()

# 注册算法
if __name__ == "__main__":
    QgsApplication.setPrefixPath("/usr", True)
    app = QgsApplication([], False)
    app.initQgis()

    processing.Processing.initialize()
    QgsApplication.processingRegistry().addProvider(QgsProcessingProvider())

    feedback = QgsProcessingFeedback()
    context = QgsProcessingContext()

    params = {
        'INPUT_FOLDER': 'path_to_input_folder',
        'TO_EPSG_4326': True,
        'OUTPUT_TIF': 'path_to_output_folder/merge.tif',
        'KEEP_VRT': False,
        'OUTPUT_TYPE': 'Float32',
        'COMPRESSION': 'LZW'
    }

    alg = MergeTiffAlgorithm()
    result = alg.processAlgorithm(params, context, feedback)
    print(result)

    app.exitQgis()
    app.exit()
