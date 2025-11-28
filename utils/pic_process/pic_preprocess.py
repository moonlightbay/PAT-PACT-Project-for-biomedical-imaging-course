"""
描述：对原始图像进行预处理，实现批量读取、缩放、索引前缀重命名
"""
import os
import cv2
import numpy as np
from typing import List
import glob

class PicPreprocess:
    def __init__(self):
        pass

    def pics_rescale(self, fromDir: str, toDir: str, scaleFactor: float, startIdx: int, endIdx: int):
        """
        按索引顺序读取文件夹下的图片，对每张图片进行缩放处理，并保存到指定文件夹中。
        文件命名格式为：prefix_0001.png, prefix_0002.png,其中0001, 0002为索引号，前缀prefix保持不变。
        缩放规则：不改变图像大小，先按照scaleFactor缩放图像，然后将图像用黑色填充到原始大小。
        - 参数:
            - fromDir: 原始图片文件夹路径
            - toDir: 处理后图片保存文件夹路径
            - scaleFactor: 缩放因子，例如0.8表示缩小到80%
            - startIdx: 起始索引号（包含）
            - endIdx: 结束索引号（包含）
        """
        # 创建目标文件夹
        os.makedirs(toDir, exist_ok=True)
        
        # 获取所有图片文件并按索引排序
        image_files = sorted(glob.glob(os.path.join(fromDir, "*.*")))
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        # 确保索引范围有效
        startIdx = max(0, startIdx)
        endIdx = min(len(image_files) - 1, endIdx)
        
        for i in range(startIdx, endIdx + 1):
            if i >= len(image_files):
                break
                
            img_path = image_files[i]
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图片: {img_path}")
                continue
            
            # 获取原始尺寸
            h, w = img.shape[:2]
            
            # 计算缩放后的尺寸
            new_h, new_w = int(h * scaleFactor), int(w * scaleFactor)
            
            # 缩放图像
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 创建黑色背景的原始尺寸画布
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            
            # 计算居中位置
            start_y = (h - new_h) // 2
            start_x = (w - new_w) // 2
            
            # 将缩放后的图像居中放置
            canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized_img
            
            # 提取原始文件名前缀
            original_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # 保存处理后的图像
            output_filename = f"{original_name}_{i+1:04d}.png"
            output_path = os.path.join(toDir, output_filename)
            cv2.imwrite(output_path, canvas)
            
            print(f"已处理并保存: {output_path}")
    
    def pics_rename(self, fromDir: str, toDir: str, prefix: str):
        """
        按文件名顺序读取文件夹下的图片，重命名为指定前缀加上四位索引号，并保存到指定文件夹中。
        文件命名格式为：prefix_0001.png, prefix_0002.png,
        其中0001, 0002为索引号，前缀prefix由用户指定。
        - 参数:
            - fromDir: 原始图片文件夹路径
            - toDir: 处理后图片保存文件夹路径
            - prefix: 文件名前缀
        """
        # 创建目标文件夹
        os.makedirs(toDir, exist_ok=True)
        
        # 获取所有图片文件并按文件名排序
        image_files = sorted(glob.glob(os.path.join(fromDir, "*.*")))
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        for i, img_path in enumerate(image_files):
            # 读取原始图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图片: {img_path}")
                continue
            
            # 生成新的文件名
            output_filename = f"{prefix}_{i+1:04d}.png"
            output_path = os.path.join(toDir, output_filename)
            
            # 保存图像
            cv2.imwrite(output_path, img)
            
            print(f"已重命名并保存: {output_path}")




