import os
import time
from samgeo import SamGeo
import json
import traceback

input_base_folder = r'D:\Dataset_and_Demo'  # 监控的文件夹路径
output_base_folder = r'D:\Dataset_and_Demo\Processed'  # 输出文件夹路径
exclude_folders = ['DataFused', ]  # 需要排除的子文件夹

box_fromGEE = True  # 是否从 GEE 获得 box

def get_files_to_process(base_folder, exclude_folders):
    '''获取需要处理的图像路径'''
    files_to_process = []
    for root, dirs, files in os.walk(base_folder):
        # 跳过排除的文件夹
        dirs[:] = [d for d in dirs if d not in exclude_folders]
        
        files = [i for i in files if '_Trans' in i]  # 过滤
        for file in files:
            if file.endswith('_ADMeanFused_Trans.tif') or file.endswith('_ADMeanFused_WithTiles_Trans.tif'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_base_folder, os.path.relpath(input_path, base_folder))
                output_path = output_path.replace('_ADMeanFused_Trans.tif', '_ADMeanFused_SAM.tif').\
                                          replace('_ADMeanFused_WithTiles_Trans.tif', '_ADMeanFused_SAM.tif')
                if not os.path.exists(output_path):
                    files_to_process.append((input_path, output_path))
    return files_to_process

def process_file(file_paths):
    try:
        input_path, output_path = file_paths
        print(f"Processing file: {input_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 初始化 SamGeo
        sam = SamGeo(
            model_type="vit_h",
            automatic=False,
            sam_kwargs=None,
        )
        sam.set_image(input_path)

        # 读取 AOI_Bound_Info 信息
        json_path = input_path.replace('_Trans.tif', '_AOI.json').replace('_Trans.tif', '_AOI.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                AOI_Bound_Info = json.load(f)
        else:
            print(f'Error: {json_path} not found.')
            return

        if box_fromGEE:
            boxes = [*AOI_Bound_Info['coordinates'][0][0], *AOI_Bound_Info['coordinates'][0][2]]
            sam.predict(boxes=boxes, point_crs="EPSG:4326", output=output_path, dtype="uint8")

    except Exception as e:
        print(f'预测时发生错误: {e}')
        with open('log.txt', 'a') as f:
            f.write(f'Predict error file = {input_path}\n')
            f.write(traceback.format_exc())
            f.write('\n')
        print(f'预测错误已记录到log.txt: {e}')

def monitor_folder():
    processed_files = set()
    while True:
        files_to_process = get_files_to_process(input_base_folder, exclude_folders)
        for file_paths in files_to_process:
            if file_paths not in processed_files:
                process_file(file_paths)
                processed_files.add(file_paths)
        time.sleep(10)  # 等待 10 秒后再检查新文件

if __name__ == "__main__":
    monitor_folder()
