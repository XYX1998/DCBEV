import os
import shutil
import json

# JSON文件路径
json_file_path = '/home/XYX/HFT-main/HFT/data/kitti_processed/kitti_360/kitti360_panopticbev/img/front.json'

# 基础源目录和目标目录
base_dir = '/home/XYX/HFT-main/HFT/data/kitti_processed/kitti_360'
target_dir = '/home/XYX/HFT-main/HFT/data/kitti360/img_dir/train'

# 相机内参矩阵
intr_00 = [
    [788.629315, 0.000000, 687.158398],
    [0.000000, 786.382230, 317.752196],
    [0.000000, 0.000000, 1.000000]
]

intr_01 = [
    [785.134093, 0.000000, 686.437073],
    [0.000000, 782.346289, 321.352788],
    [0.000000, 0.000000, 1.000000]
]

# 读取JSON文件
with open(json_file_path, 'r') as file:
    json_data = json.load(file)

# 确保目标目录存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 创建一个字典来存储新文件名和对应的相机内参
calib_dict = {}

# 遍历JSON数据中的每个条目
for item in json_data:
    for new_filename, relative_path in item.items():
        # 构建完整源文件路径
        source_file_path = os.path.join(base_dir, relative_path)
        
        # 目标文件路径
        target_file_path = os.path.join(target_dir, new_filename)
        
        try:
            # 移动并重命名文件
            shutil.copy(source_file_path, target_file_path)
            print(f"copy and renamed: {source_file_path} to {target_file_path}")
            
            # 根据路径决定使用哪个相机内参
            if 'image_00' in relative_path:
                calib_dict[new_filename] = intr_00
            elif 'image_01' in relative_path:
                calib_dict[new_filename] = intr_01
            
        except FileNotFoundError:
            print(f"File not found: {source_file_path}")
        except Exception as e:
            print(f"Error moving file {source_file_path} to {target_file_path}: {e}")

# 保存相机内参信息到新的JSON文件
calib_json_path = '/home/XYX/HFT-main/HFT/data/kitti360/calib.json'
with open(calib_json_path, 'w') as calib_file:
    json.dump(calib_dict, calib_file, indent=4)

print("操作完成")