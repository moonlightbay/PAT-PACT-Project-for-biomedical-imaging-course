#!/usr/bin/env python3
"""
根据PACT仿真配置计算合理的图像缩放因子
仿真参数：
- 网格尺寸：256x256
- 网格间距：0.1mm
- 传感器半径：(256/2 - 10) * 0.1mm = 11.8mm
"""

import os
import cv2
import numpy as np
import glob

def analyze_image_dimensions():
    """分析MRA图像的尺寸"""
    mra_dir = "data/raw/output_MRA"
    
    if not os.path.exists(mra_dir):
        print(f"目录不存在: {mra_dir}")
        return None
    
    # 获取所有PNG图像
    image_files = glob.glob(os.path.join(mra_dir, "*.png"))
    
    if not image_files:
        print(f"在 {mra_dir} 中未找到PNG图像")
        return None
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 分析前几张图像的尺寸
    dimensions = []
    for img_path in image_files[:5]:  # 分析前5张
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape
            dimensions.append((h, w))
            print(f"图像 {os.path.basename(img_path)}: {w}x{h}")
    
    if not dimensions:
        print("无法读取任何图像")
        return None
    
    # 计算统计信息
    widths = [d[0] for d in dimensions]
    heights = [d[1] for d in dimensions]
    
    avg_width = np.mean(widths)
    avg_height = np.mean(heights)
    max_dim = max(avg_width, avg_height)
    
    print(f"\n尺寸统计:")
    print(f"平均宽度: {avg_width:.1f} 像素")
    print(f"平均高度: {avg_height:.1f} 像素")
    print(f"最大维度: {max_dim:.1f} 像素")
    
    return {
        'avg_width': avg_width,
        'avg_height': avg_height,
        'max_dim': max_dim,
        'sample_dims': dimensions
    }

def calculate_scale_factor(image_stats):
    """根据仿真配置计算缩放因子"""
    if image_stats is None:
        return None
    
    # 仿真参数
    grid_size = 256  # 256x256网格
    sensor_radius_mm = 11.8  # 传感器半径 11.8mm
    
    # 假设原始图像中的1个像素对应现实中的某个物理尺寸
    # 对于MRA图像，通常1像素 ≈ 0.5-1mm（取决于具体的MRA序列）
    # 这里我们采用保守估计：1像素 ≈ 0.6mm
    
    pixels_per_mm = 1.0 / 0.6  # 约1.67像素/mm
    effective_diameter_mm = 2 * sensor_radius_mm  # 传感器有效直径 = 23.6mm
    
    # 计算图像应该缩放到的像素尺寸
    target_size_pixels = effective_diameter_mm * pixels_per_mm
    target_size_pixels = int(target_size_pixels)  # 约17.7像素
    
    # 但是我们不能缩放到这么小，否则图像信息损失太大
    # 实际上，我们应该确保血管结构在传感器范围内，而不是整个图像
    
    # 更合理的方法：假设血管结构占据图像的60-80%
    vessel_coverage = 0.7  # 假设血管占据图像的70%
    effective_image_size_mm = image_stats['max_dim'] / pixels_per_mm * vessel_coverage
    
    # 计算缩放因子
    current_max_dim = image_stats['max_dim']
    
    # 方法1：基于传感器直径计算
    scale_factor_method1 = target_size_pixels / current_max_dim
    
    # 方法2：基于经验公式 (更保守)
    # 确保缩放后图像大小合适，建议最终图像大小为180-220像素
    recommended_final_size = 200  # 像素
    scale_factor_method2 = recommended_final_size / current_max_dim
    
    # 方法3：基于物理尺寸
    # 假设原始图像覆盖约50mmx50mm的区域
    original_coverage_mm = 50  # 假设原始MRA覆盖50mm
    scale_factor_method3 = effective_diameter_mm / original_coverage_mm
    
    print(f"\n缩放因子计算:")
    print(f"当前最大维度: {current_max_dim:.1f} 像素")
    print(f"传感器有效直径: {effective_diameter_mm:.1f} mm")
    print(f"方法1 (基于传感器直径): {scale_factor_method1:.3f}")
    print(f"方法2 (基于推荐尺寸): {scale_factor_method2:.3f}")
    print(f"方法3 (基于物理尺寸): {scale_factor_method3:.3f}")
    
    # 推荐使用保守的方法2
    recommended_scale = scale_factor_method2
    print(f"\n推荐缩放因子: {recommended_scale:.3f}")
    
    # 验证缩放后的尺寸
    final_size = current_max_dim * recommended_scale
    print(f"缩放后最大维度: {final_size:.1f} 像素")
    print(f"相比256网格的比例: {final_size/256:.2f}")
    
    return recommended_scale

def main():
    print("=== PACT图像缩放因子计算 ===\n")
    
    # 分析图像尺寸
    image_stats = analyze_image_dimensions()
    
    if image_stats is None:
        return
    
    # 计算缩放因子
    scale_factor = calculate_scale_factor(image_stats)
    
    if scale_factor is not None:
        print(f"\n=== 最终建议 ===")
        print(f"推荐缩放因子: {scale_factor:.3f}")
        print(f"\n在pic_preprocess.py中使用:")
        print(f"processor.pics_rescale(")
        print(f"    fromDir='data/raw/output_MRA',")
        print(f"    toDir='data/processed/vessels',")
        print(f"    scaleFactor={scale_factor:.3f},")
        print(f"    startIdx=0,")
        print(f"    endIdx=9")
        print(f")")

if __name__ == "__main__":
    main()