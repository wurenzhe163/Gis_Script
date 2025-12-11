import ee
import geemap
import numpy as np
import rasterio
import os
import argparse

# 初始化 GEE
try:
    ee.Initialize(project='ee-mrwurenzhe')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='ee-mrwurenzhe')

def download_dem(
    region,
    dataset='COPERNICUS/DEM/GLO30',
    scale=30,
    output_dir='dem_output',
    generate_binary=True,
    generate_par=True
):
    """
    下载 Google Earth Engine DEM 数据，针对单个经纬度范围。

    参数:
        region (list): 包含 [min_lon, max_lon, min_lat, max_lat] 的列表
        dataset (str): GEE DEM 数据集 ID（默认 'COPERNICUS/DEM/GLO30'）
        scale (float): 分辨率（米，默认 30）
        output_dir (str): 输出文件目录（默认 'dem_output'）
        generate_binary (bool): 是否生成二进制文件（默认 True）
        generate_par (bool): 是否生成 .par 元数据文件（默认 True）

    返回:
        dict: 包含下载结果的字典
    """
    # 有效数据集
    VALID_DATASETS = {
        'COPERNICUS/DEM/GLO30': {'type': 'ImageCollection', 'band': 'DEM'},
        'NASA/NASADEM_HGT/001': {'type': 'Image', 'band': 'elevation'},
        'JAXA/ALOS/AW3D30/V3_2': {'type': 'ImageCollection', 'band': 'DSM'}
    }

    # 验证数据集
    if dataset not in VALID_DATASETS:
        raise ValueError(f"无效的数据集: {dataset}。支持的数据集: {list(VALID_DATASETS.keys())}")

    # 验证区域格式
    if len(region) != 4:
        raise ValueError(f"区域格式错误: {region}，应为 [min_lon, max_lon, min_lat, max_lat]")

    min_lon, max_lon, min_lat, max_lat = region
    print(f"处理区域: [{min_lon}, {max_lon}, {min_lat}, {max_lat}]")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 定义研究区域
    ee_region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    # 选择并裁剪 DEM 数据集
    try:
        dataset_info = VALID_DATASETS[dataset]
        if dataset_info['type'] == 'ImageCollection':
            dataset_obj = ee.ImageCollection(dataset).mosaic().select(dataset_info['band']).rename('elevation')
        else:
            dataset_obj = ee.Image(dataset).select(dataset_info['band']).rename('elevation')
        dem_clipped = dataset_obj.clip(ee_region)
    except Exception as e:
        print(f"加载数据集失败: {str(e)}")
        return {
            'region': region,
            'geotiff_file': None,
            'binary_file': None,
            'par_file': None,
            'metadata': {}
        }

    # 设置输出文件名
    dataset_name = dataset.replace('/', '_')
    geotiff_file = os.path.join(output_dir, f"{dataset_name}.tif")
    binary_file = os.path.join(output_dir, f"{dataset_name}.bin")
    par_file = os.path.join(output_dir, f"{dataset_name}.par")

    # 直接下载到本地
    try:
        print(f"正在下载到 {geotiff_file}...")
        geemap.download_ee_image(
            image=dem_clipped,
            filename=geotiff_file,
            region=ee_region,
            scale=scale,
            crs='EPSG:4326'
        )
        print(f"GeoTIFF 文件已保存为: {geotiff_file}")
    except Exception as e:
        print(f"下载失败: {str(e)}")
        return {
            'region': region,
            'geotiff_file': None,
            'binary_file': None,
            'par_file': None,
            'metadata': {}
        }

    # 处理 GeoTIFF 文件以生成二进制和元数据文件
    metadata = {}
    if generate_binary or generate_par:
        try:
            with rasterio.open(geotiff_file) as src:
                dem_data = src.read(1).astype(np.float32)
                profile = src.profile
                metadata = {
                    'width': dem_data.shape[1],
                    'height': dem_data.shape[0],
                    'min_lon': min_lon,
                    'max_lon': max_lon,
                    'min_lat': min_lat,
                    'max_lat': max_lat,
                    'resolution': scale / 111000,
                    'crs': str(profile['crs']),
                    'transform': profile['transform']
                }

            # 保存二进制文件
            if generate_binary:
                dem_data.tofile(binary_file)
                print(f"二进制文件已保存为: {binary_file}")

            # 生成 .par 文件
            if generate_par:
                with open(par_file, 'w') as f:
                    f.write(f"width: {metadata['width']}\n")
                    f.write(f"height: {metadata['height']}\n")
                    f.write(f"min_lon: {metadata['min_lon']}\n")
                    f.write(f"max_lon: {metadata['max_lon']}\n")
                    f.write(f"min_lat: {metadata['min_lat']}\n")
                    f.write(f"max_lat: {metadata['max_lat']}\n")
                    f.write(f"resolution: {metadata['resolution']}\n")
                    f.write(f"crs: {metadata['crs']}\n")
                print(f"元数据文件已保存为: {par_file}")
        except Exception as e:
            print(f"处理 GeoTIFF 文件失败: {str(e)}")
            return {
                'region': region,
                'geotiff_file': geotiff_file,
                'binary_file': None,
                'par_file': None,
                'metadata': {}
            }

    # 可视化（可选）
    try:
        import matplotlib.pyplot as plt
        plt.imshow(dem_data, cmap='terrain')
        plt.colorbar()
        plt.axis('image')
        plt.title(f"DEM: {dataset_name}")
        plt.show()
    except ImportError:
        print("Matplotlib 未安装，跳过可视化。")

    return {
        'region': region,
        'geotiff_file': geotiff_file,
        'binary_file': binary_file if generate_binary else None,
        'par_file': par_file if generate_par else None,
        'metadata': metadata
    }

def parse_region(s):
    """
    解析命令行输入的区域字符串，格式为 'min_lon,max_lon,min_lat,max_lat'。
    """
    try:
        min_lon, max_lon, min_lat, max_lat = map(float, s.split(','))
        return [min_lon, max_lon, min_lat, max_lat]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"区域格式错误: {s}，应为 'min_lon,max_lon,min_lat,max_lat'")

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="下载 Google Earth Engine DEM 数据")
    parser.add_argument(
        '--region',
        type=parse_region,
        required=True,
        help="经纬度范围，格式为 'min_lon,max_lon,min_lat,max_lat'"
    )
    parser.add_argument(
        '--dataset',
        default='COPERNICUS/DEM/GLO30',
        choices=[
            'COPERNICUS/DEM/GLO30',
            'NASA/NASADEM_HGT/001',
            'JAXA/ALOS/AW3D30/V3_2'
        ],
        help="GEE DEM 数据集（默认: COPERNICUS/DEM/GLO30）"
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=30,
        help="分辨率（米，默认: 30）"
    )
    parser.add_argument(
        '--output-dir',
        default='dem_output',
        help="输出文件目录（默认: dem_output）"
    )
    parser.add_argument(
        '--no-binary',
        action='store_false',
        dest='generate_binary',
        help="不生成二进制文件"
    )
    parser.add_argument(
        '--no-par',
        action='store_false',
        dest='generate_par',
        help="不生成 .par 元数据文件"
    )

    args = parser.parse_args()

    # 执行下载
    result = download_dem(
        region=args.region,
        dataset=args.dataset,
        scale=args.scale,
        output_dir=args.output_dir,
        generate_binary=args.generate_binary,
        generate_par=args.generate_par
    )

    # 打印结果
    print("\n下载结果:")
    print(f"区域: {result['region']}")
    print(f"GeoTIFF 文件: {result['geotiff_file']}")
    print(f"二进制文件: {result['binary_file']}")
    print(f"元数据文件: {result['par_file']}")
    print(f"元数据: {result['metadata']}")

if __name__ == "__main__":
    main()