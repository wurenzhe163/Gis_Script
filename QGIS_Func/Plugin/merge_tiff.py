import os
from osgeo import gdal
from qgis.PyQt.QtCore import QCoreApplication, QObject
from qgis.PyQt.QtWidgets import QAction, QFileDialog, QMessageBox
from qgis.core import QgsProject

class MergeTIFFPlugin(QObject):
    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = self.tr("&Merge TIFF Plugin")
        self.first_start = None

    def tr(self, message):
        return QCoreApplication.translate('MergeTIFFPlugin', message)

    def add_action(self, icon_path, text, callback, parent=None):
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        self.iface.addPluginToMenu(self.menu, action)
        self.actions.append(action)
        return action

    def initGui(self):
        icon_path = ':/plugins/merge_tiff_plugin/icon.png'
        self.add_action(icon_path, self.tr("Merge TIFF Images"), self.run, self.iface.mainWindow())

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(self.tr("&Merge TIFF Plugin"), action)
        self.actions = []

    def run(self):
        input_folder = QFileDialog.getExistingDirectory(self.iface.mainWindow(), "Select Input Folder")
        if not input_folder:
            return

        reply = QMessageBox.question(self.iface.mainWindow(), 'Transform Projection',
                                     'Do you want to transform the image projection to EPSG:4326?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        to_epsg_4326 = (reply == QMessageBox.Yes)
        process_images(input_folder, to_epsg_4326)

        QMessageBox.information(self.iface.mainWindow(), 'Success', 'TIFF images merged successfully!')

def search_files(path, endwith='.tif'):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(endwith)]

def check_projection(input_group):
    projections = []
    for file_path in input_group:
        dataset = gdal.Open(file_path)
        projections.append(dataset.GetProjection())
        dataset = None
    return projections

def process_images(input_folder, to_epsg_4326=False):
    os.chdir(input_folder)
    input_group = search_files(path=input_folder, endwith='.tif')

    projections = check_projection(input_group)

    output_vrt = 'temp.vrt'
    output_tif = 'merge.tif'

    if to_epsg_4326:
        aligned_inputs = []
        for img in input_group:
            aligned_img = img.replace('.tif', '_aligned.tif')
            gdal.Warp(aligned_img, img, dstSRS='EPSG:4326')  # 转换至 WGS84 投影
            aligned_inputs.append(aligned_img)
    else:
        aligned_inputs = input_group

    # 创建虚拟数据集
    gdal.BuildVRT(output_vrt, aligned_inputs)
    # 使用期望的重采样算法进行图像拼接
    gdal.Warp(output_tif, output_vrt, resampleAlg='bilinear') #'average', 'nearest', 'bilinear', 'cubic', 'cubicspline', 'lanczos', 'average_magphase', 'mode'

    os.remove(output_vrt)
    if to_epsg_4326:
        for img in aligned_inputs:
            os.remove(img)
