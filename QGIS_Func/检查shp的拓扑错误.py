from qgis.core import (
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsField,
    QgsProject,
    QgsVectorFileWriter,
    QgsWkbTypes
)
from PyQt5.QtCore import QVariant
from PyQt5.QtWidgets import QInputDialog, QMessageBox
from qgis.analysis import QgsGeometryAnalyzer

def check_topology_errors(shapefile_path):
    # 读取shapefile
    layer = QgsVectorLayer(shapefile_path, "Input Layer", "ogr")
    if not layer.isValid():
        QMessageBox.critical(None, "错误", "无法读取Shapefile文件。")
        return None

    # 创建一个临时shapefile图层用于存储拓扑错误
    error_layer = QgsVectorLayer("Polygon?crs=EPSG:4326", "Topology Errors", "memory")
    error_provider = error_layer.dataProvider()

    # 添加字段
    error_provider.addAttributes([
        QgsField("ErrorType", QVariant.String),
        QgsField("ValidityReason", QVariant.String),
        QgsField("OverlapWith", QVariant.Int)
    ])
    error_layer.updateFields()

    features = layer.getFeatures()
    feature_list = list(features)

    # 检查几何对象是否有效
    for idx, feature in enumerate(feature_list):
        geom = feature.geometry()

        # 检查是否存在自相交拓扑错误
        if not geom.isGeosValid():
            new_feature = QgsFeature()
            new_feature.setGeometry(geom)
            new_feature.setAttributes(["自相交", "自相交", None])
            error_provider.addFeature(new_feature)

        # 检查重叠错误（仅适用于多边形和多重多边形）
        if geom.type() == QgsWkbTypes.PolygonGeometry:
            for other_idx, other_feature in enumerate(feature_list):
                if idx != other_idx and geom.intersects(other_feature.geometry()):
                    new_feature = QgsFeature()
                    new_feature.setGeometry(geom)
                    new_feature.setAttributes(["重叠", None, other_idx])
                    error_provider.addFeature(new_feature)

        # 检查空洞错误
        if geom.isMultipart():
            polygons = geom.asMultiPolygon()
        else:
            polygons = [geom.asPolygon()]

        for polygon in polygons:
            if len(polygon) > 1:  # 检查是否有内环
                for interior in polygon[1:]:
                    interior_geom = QgsGeometry.fromPolygonXY([interior])
                    if not interior_geom.isGeosValid():
                        new_feature = QgsFeature()
                        new_feature.setGeometry(interior_geom)
                        new_feature.setAttributes(["空洞", "空洞", None])
                        error_provider.addFeature(new_feature)

    error_layer.updateExtents()
    QgsProject.instance().addMapLayer(error_layer)

    # 保存为临时shp文件
    temp_shp_path = "/tmp/topology_errors.shp"
    QgsVectorFileWriter.writeAsVectorFormat(error_layer, temp_shp_path, "utf-8", error_layer.crs(), "ESRI Shapefile")

    return temp_shp_path

# 提示用户输入shapefile路径
shapefile_path, ok = QInputDialog.getText(None, "输入Shapefile路径", "Shapefile路径:")

if ok and shapefile_path:
    temp_shp_path = check_topology_errors(shapefile_path)
    if temp_shp_path:
        QMessageBox.information(None, "完成", f"拓扑错误已保存到 {temp_shp_path}")
else:
    QMessageBox.warning(None, "取消", "操作已取消。")
