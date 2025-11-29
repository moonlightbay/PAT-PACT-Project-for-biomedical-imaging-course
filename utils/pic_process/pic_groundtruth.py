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
                            core_center=None, core_radius=None, penumbra_width=None,
                            so2_levels=None):
        """
        基于血管掩膜生成模拟脑卒中SO2分布的图像，并可选生成对应的分割标签。
        
        参数:
            mask_path: 血管掩膜图片路径
            output_img_path: 输出模拟SO2图像路径
            output_label_path: 输出分割标签图像路径 (可选)
            core_center: (x, y) 梗死核心中心坐标，若为None则随机生成
            core_radius: 梗死核心半径，若为None则随机生成
            penumbra_width: 半暗带宽度，若为None则随机生成
            so2_levels: 字典，定义各区域的SO2值(0-255)
        """
        # 1. 读取血管掩膜
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法读取图片: {mask_path}")
            return None

        h, w = mask.shape
        
        # 2. 设置参数默认值
        if core_center is None:
            # 随机选择图像内一点作为病灶中心，优先选择血管区域（高灰度值区域）
            # 找到所有灰度值大于阈值的像素坐标
            vessel_indices = np.where(mask > 20) # 假设血管区域灰度值大于20
            if len(vessel_indices[0]) > 0:
                # 从血管像素中随机选择一个
                idx = random.randint(0, len(vessel_indices[0]) - 1)
                core_center = (vessel_indices[1][idx], vessel_indices[0][idx]) # (x, y)
            else:
                # 如果没有找到血管像素，则回退到全图随机
                margin = min(h, w) // 5
                core_center = (random.randint(margin, w-margin), random.randint(margin, h-margin))
            
        if core_radius is None:
            # 核心半径范围
            core_radius = random.randint(min(h, w)//15, min(h, w)//8)
            
        if penumbra_width is None:
            # 半暗带宽度范围
            penumbra_width = random.randint(min(h, w)//15, min(h, w)//8)
            
        if so2_levels is None:
            # 模拟SO2值：正常(高) > 半暗带(中) > 梗死(低)
            so2_levels = {
                'normal': 220,   # 正常区域
                'penumbra': 140, # 半暗带
                'core': 60       # 梗死核心
            }

        # 3. 创建距离场
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - core_center[0])**2 + (Y - core_center[1])**2)
        
        # 4. 定义区域掩膜
        core_mask = dist_from_center <= core_radius
        penumbra_mask = (dist_from_center > core_radius) & (dist_from_center <= (core_radius + penumbra_width))
        normal_mask = dist_from_center > (core_radius + penumbra_width)
        
        # 5. 生成SO2分布图 (Image)
        so2_map = np.zeros_like(mask, dtype=np.float32)
        
        # 基础血管掩膜二值化 (0或1)
        _, vessel_binary = cv2.threshold(mask, 10, 1, cv2.THRESH_BINARY)
        
        # 赋予各区域SO2值
        so2_map[normal_mask] = so2_levels['normal']
        
        # 半暗带区域渐变处理 (可选，这里使用线性渐变模拟生理过渡)
        # 距离核心越近越低，越远越高
        penumbra_dists = dist_from_center[penumbra_mask]
        if len(penumbra_dists) > 0:
            # 归一化距离 0~1
            norm_dists = (penumbra_dists - core_radius) / penumbra_width
            # 线性插值
            penumbra_vals = so2_levels['core'] + norm_dists * (so2_levels['normal'] - so2_levels['core'])
            # 添加一点随机噪声模拟真实情况
            noise = np.random.normal(0, 5, penumbra_vals.shape)
            so2_map[penumbra_mask] = np.clip(penumbra_vals + noise, so2_levels['core'], so2_levels['normal'])
        else:
            so2_map[penumbra_mask] = so2_levels['penumbra']
            
        so2_map[core_mask] = so2_levels['core']
        
        # 应用血管掩膜
        final_img = so2_map * vessel_binary
        final_img = final_img.astype(np.uint8)
        
        # 保存SO2图像
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        cv2.imwrite(output_img_path, final_img)
        
        # 6. 生成分割标签 (Label) - 如果需要
        if output_label_path:
            # 定义标签值: 0-背景, 1-正常, 2-半暗带, 3-梗死
            label_map = np.zeros_like(mask, dtype=np.uint8)
            
            # 只有在血管区域内才标记
            label_map[normal_mask & (vessel_binary==1)] = 1
            label_map[penumbra_mask & (vessel_binary==1)] = 2
            label_map[core_mask & (vessel_binary==1)] = 3
            
            os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
            # 保存为png，注意像素值很小(0,1,2,3)，肉眼看是全黑的
            cv2.imwrite(output_label_path, label_map)
            
        return {
            'core_center': core_center,
            'core_radius': core_radius,
            'penumbra_width': penumbra_width
        }

    def batch_process(self, input_dir, output_img_dir, output_label_dir=None, count_per_image=1):
        """
        批量处理文件夹下的所有图片
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
        output_label_path="test/simulated_so2_images/sample_label.png"
    )


def test_batch_generate():
    pg = PicGroundTruth()
    pg.batch_process(
        input_dir="data/mask_images",
        output_img_dir="data/simulated_so2_images",
        output_label_dir=None,
        count_per_image=1
    )   

if __name__ == "__main__":
    test_single_generate()