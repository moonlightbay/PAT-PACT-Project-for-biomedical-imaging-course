"""
描述: 修改预处理后的血管掩膜图像的像素值，模拟脑卒中患者的SO2分布情况，作为模型训练的真实标签数据。
"""
import cv2
import numpy as np
import random
import os
import glob

class PicGroundTruth:
    def __init__(self):
        pass

    def generate_stroke_so2(self, mask_path, output_img_path, output_label_path=None,
                            num_lesions_range=(1, 3), min_lesion_dist=None,
                            core_radius_range=None, penumbra_width_range=None,
                            so2_ranges=None):
        """
        基于血管掩膜生成模拟脑卒中SO2分布的图像，支持多病灶，并可选生成对应的分割标签。
        
        参数:
            mask_path: 血管掩膜图片路径
            output_img_path: 输出模拟SO2图像路径
            output_label_path: 输出分割标签图像路径 (可选)
            num_lesions_range: (min, max) 随机生成的病灶数量范围
            min_lesion_dist: 多个病灶之间的最小距离，若为None则自动计算
            core_radius_range: (min, max) 梗死核心半径范围
            penumbra_width_range: (min, max) 半暗带宽度范围
            so2_ranges: 字典，定义各区域的SO2值范围 {'normal': (min, max), 'penumbra': (min, max), 'core': (min, max)}
        """
        # 1. 读取血管掩膜
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法读取图片: {mask_path}")
            return None

        h, w = mask.shape
        min_dim = min(h, w)
        
        # 2. 设置参数默认值
        if min_lesion_dist is None:
            min_lesion_dist = min_dim // 6
            
        if core_radius_range is None:
            core_radius_range = (min_dim // 20, min_dim // 12)
            
        if penumbra_width_range is None:
            penumbra_width_range = (min_dim // 10, min_dim // 6)
            
        if so2_ranges is None:
            # 模拟SO2值范围
            so2_ranges = {
                'normal': (210, 255),   # 正常区域
                'penumbra': (110, 160), # 半暗带 (作为渐变的参考值)
                'core': (30, 60)        # 梗死核心
            }

        # 3. 初始化全图 SO2 和 Label
        # 全图基准正常值
        base_normal_so2 = random.randint(*so2_ranges['normal'])
        # 初始化为基准值
        final_so2_map = np.full((h, w), base_normal_so2, dtype=np.float32)
        # 添加全图基础噪声
        final_so2_map += np.random.normal(0, 3, (h, w))
        
        # 初始化 Label (1: Normal)
        # 0: Background (masked out later), 1: Normal, 2: Penumbra, 3: Core
        final_label_map = np.ones((h, w), dtype=np.uint8)

        # 4. 确定病灶数量和中心
        num_lesions = random.randint(*num_lesions_range)
        centers = []
        
        # 找到所有血管像素
        vessel_indices = np.where(mask > 20)
        has_vessel = len(vessel_indices[0]) > 0
        
        # 生成第一个中心
        if has_vessel:
            idx = random.randint(0, len(vessel_indices[0]) - 1)
            center1 = (vessel_indices[1][idx], vessel_indices[0][idx])
        else:
            margin = min_dim // 5
            center1 = (random.randint(margin, w-margin), random.randint(margin, h-margin))
        centers.append(center1)
        
        # 生成后续中心 (彼此之间保持一定距离)
        for _ in range(num_lesions - 1):
            new_center = None
            
            # 尝试寻找合适的血管像素
            if has_vessel:
                # 计算所有血管像素到现有所有中心的距离
                dists_list = []
                for c in centers:
                    d = np.sqrt((vessel_indices[1] - c[0])**2 + (vessel_indices[0] - c[1])**2)
                    dists_list.append(d)
                
                # 计算每个像素到最近中心的距离
                if dists_list:
                    min_dists = np.min(dists_list, axis=0)
                    # 筛选出距离大于最小距离的像素
                    valid_indices_mask = min_dists >= min_lesion_dist
                    valid_indices_idx = np.where(valid_indices_mask)[0]
                    
                    if len(valid_indices_idx) > 0:
                        idx = valid_indices_idx[random.randint(0, len(valid_indices_idx) - 1)]
                        new_center = (vessel_indices[1][idx], vessel_indices[0][idx])
            
            # 如果没找到合适的血管像素，或者没有血管，就在全图随机选一个点
            if new_center is None:
                for _ in range(50): # 尝试50次
                    margin = min_dim // 10
                    cx = random.randint(margin, w-margin)
                    cy = random.randint(margin, h-margin)
                    
                    # 检查距离
                    too_close = False
                    for c in centers:
                        dist = np.sqrt((cx - c[0])**2 + (cy - c[1])**2)
                        if dist < min_lesion_dist:
                            too_close = True
                            break
                    
                    if not too_close:
                        new_center = (cx, cy)
                        break
            
            if new_center is not None:
                centers.append(new_center)

        # 5. 逐个生成病灶并融合
        Y, X = np.ogrid[:h, :w]
        
        for center in centers:
            # 随机生成当前病灶的参数
            r_core = random.randint(*core_radius_range)
            w_penumbra = random.randint(*penumbra_width_range)
            val_core = random.randint(*so2_ranges['core'])
            
            # 计算距离场
            dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
            
            # 定义区域掩膜
            core_mask = dist_from_center <= r_core
            penumbra_mask = (dist_from_center > r_core) & (dist_from_center <= (r_core + w_penumbra))
            
            # 生成当前病灶的 SO2 分布
            # 初始化为无穷大，方便取 min
            lesion_so2 = np.full((h, w), float('inf'), dtype=np.float32)
            
            # 核心区域
            lesion_so2[core_mask] = val_core
            
            # 半暗带区域 (线性渐变: val_core -> base_normal_so2)
            penumbra_dists = dist_from_center[penumbra_mask]
            if len(penumbra_dists) > 0:
                norm_dists = (penumbra_dists - r_core) / w_penumbra
                # 渐变到全图基准正常值，保证边缘平滑
                penumbra_vals = val_core + norm_dists * (base_normal_so2 - val_core)
                noise = np.random.normal(0, 5, penumbra_vals.shape)
                lesion_so2[penumbra_mask] = penumbra_vals + noise
            
            # 融合 SO2 (取最小值，模拟缺血叠加)
            # 只有在病灶范围内才更新，避免 inf 覆盖
            affected_mask = core_mask | penumbra_mask
            final_so2_map[affected_mask] = np.minimum(final_so2_map[affected_mask], lesion_so2[affected_mask])
            
            # 融合 Label (取最大值: 3 Core > 2 Penumbra > 1 Normal)
            final_label_map[penumbra_mask] = np.maximum(final_label_map[penumbra_mask], 2)
            final_label_map[core_mask] = np.maximum(final_label_map[core_mask], 3)

        # 6. 应用血管掩膜和后处理
        # 基础血管掩膜二值化
        _, vessel_binary = cv2.threshold(mask, 10, 1, cv2.THRESH_BINARY)
        
        # 限制 SO2 范围
        final_so2_map = np.clip(final_so2_map, 0, 255)
        
        # 应用掩膜
        final_img = final_so2_map * vessel_binary
        final_img = final_img.astype(np.uint8)
        
        # 保存SO2图像
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        cv2.imwrite(output_img_path, final_img)
        
        # 7. 生成分割标签
        if output_label_path:
            # 背景设为 0
            final_label_map = final_label_map * vessel_binary
            
            os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
            cv2.imwrite(output_label_path, final_label_map)
            
        return {
            'centers': centers,
            'num_lesions': num_lesions
        }

    def batch_process(self, input_dir, output_img_dir, output_label_dir=None, count_per_image=1):
        """
        批量处理文件夹下的所有图片
        args:
            input_dir: 输入图片文件夹路径
            output_img_dir: 输出模拟SO2图片文件夹路径
            output_label_dir: 输出分割标签文件夹路径 (可选)
            count_per_image: 每张输入图片生成的模拟图片数量
        """
        image_files = sorted(glob.glob(os.path.join(input_dir, "*.*")))
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"找到 {len(image_files)} 张图片，开始处理...")
        
        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            for i in range(count_per_image):
                # 构建输出路径
                suffix = f"_sim_{i+1:02d}" if count_per_image > 1 else ""
                
                out_img_name = f"{base_name}{suffix}.png"
                out_img_path = os.path.join(output_img_dir, out_img_name)
                
                out_label_path = None
                if output_label_dir:
                    out_label_name = f"{base_name}{suffix}_label.png"
                    out_label_path = os.path.join(output_label_dir, out_label_name)
                
                # 生成
                self.generate_stroke_so2(img_path, out_img_path, out_label_path)
                
        print("批量处理完成。")

def test_single_generate():
    pg = PicGroundTruth()
    pg.generate_stroke_so2(
        mask_path="data/raw/output_MRA/10001-TOF_ADAM_mip.png",
        output_img_path="test/simulated_so2_images/sample_so2.png",
        output_label_path="test/simulated_so2_images/sample_label.png",
        num_lesions_range=(1, 3)
    )


def test_batch_generate():
    pg = PicGroundTruth()
    pg.batch_process(
        input_dir="data/processed/renamed/",
        output_img_dir="data/mask/",
        output_label_dir="data/label/",
        count_per_image=1
    )   

if __name__ == "__main__":
    test_batch_generate()  