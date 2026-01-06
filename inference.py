# 简单推理脚本
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import os
import argparse
import glob

# 导入项目中的自定义模块
from models.uad import ViTill
from models import vit_encoder
from utils import cal_anomaly_maps, cvt2heatmap, show_cam_on_image, get_gaussian_kernel
from dataset import get_data_transforms


def load_model(model_path, device):
    """加载训练好的模型"""
    # 模型配置参数（需与训练时保持一致）
    encoder_name = 'dinov2reg_vit_base_14'
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # 构建编码器
    encoder = vit_encoder.load(encoder_name)

    # 根据编码器类型设置嵌入维度和头数
    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("Architecture not in small, base, large.")

    # 构建模型
    bottleneck = []
    decoder = []

    # 从models.vision_transformer导入必要的类
    from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
    from functools import partial

    # 构建bottleneck
    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
    bottleneck = torch.nn.ModuleList(bottleneck)

    # 构建decoder
    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-8),
                       attn=LinearAttention2)
        decoder.append(blk)
    decoder = torch.nn.ModuleList(decoder)

    # 构建完整模型
    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                   target_layers=target_layers, mask_neighbor_size=0,
                   fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model


def preprocess_image(image_path, device):
    """预处理输入图像"""
    # 与训练时相同的图像尺寸
    image_size = 1022
    crop_size = 1022

    # 获取数据转换
    data_transform, _ = get_data_transforms(image_size, crop_size)

    # 加载并转换图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size

    # 应用转换
    image_tensor = data_transform(image).unsqueeze(0).to(device)

    return image_tensor, original_size


def simple_infer(model, image_tensor, device, use_normalization=True):
    """执行简单推理，参考evaluation函数实现"""
    model.eval()

    with torch.no_grad():
        # 模型前向传播
        en, de = model(image_tensor)

        # 计算异常图（使用返回PyTorch张量的cal_anomaly_maps函数）
        anomaly_map, _ = cal_anomaly_maps(en, de, image_tensor.shape[-1])

        # 应用高斯滤波
        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
        anomaly_map = gaussian_kernel(anomaly_map)[0, 0]

        # 图像级异常分数（最大异常值）
        img_score = torch.max(anomaly_map)

        # 根据开关决定是否归一化异常图
        if use_normalization:
            # 归一化异常图用于可视化
            anomaly_map_processed = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
        else:
            # 不进行归一化，直接返回原始异常图
            anomaly_map_processed = anomaly_map

    return anomaly_map_processed.cpu().numpy(), img_score.item()


def visualize_results(image_path, anomaly_map, output_dir, use_normalization=True, threshold=None):
    """可视化结果"""
    # 加载原始图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 调整异常图大小以匹配原始图像
    anomaly_map_resized = cv2.resize(anomaly_map, (image.shape[1], image.shape[0]))


    # 保存结果目录
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # 保存原始图像
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_original.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if use_normalization:
        # 生成热图
        heatmap = cvt2heatmap((anomaly_map_resized * 255).astype(np.uint8))
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_heatmap.png'), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    else:
        # 生成二值化图
        if threshold is None:
            # 如果没有指定阈值，使用异常图的中位数作为阈值
            threshold = np.median(anomaly_map_resized)
        
        # 二值化处理，异常分数大于阈值的置为255，否则为0
        binary_map = (anomaly_map_resized > threshold).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_binary.png'), binary_map)



def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='简单异常检测推理脚本')
    parser.add_argument('--image_path', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--output_dir', type=str, default='./infer_result', help='结果输出目录')
    parser.add_argument(
        '--no_normalization',
        action='store_false',
        dest='use_normalization',
        help='关闭异常图归一化'
    )
    parser.add_argument('--threshold', type=float, default=None, help='二值化阈值，不指定则使用中位数')
    args = parser.parse_args()

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载模型
    print("正在加载模型...")
    model = load_model(args.model_path, device)

    # 确定要处理的图像列表
    if os.path.isfile(args.image_path):
        # 单个图像文件
        image_files = [args.image_path]
    elif os.path.isdir(args.image_path):
        # 文件夹，获取所有图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
            image_files.extend(glob.glob(os.path.join(args.image_path, ext)))
        if not image_files:
            print(f"在目录 {args.image_path} 中未找到图像文件")
            return
    else:
        print(f"输入路径 {args.image_path} 不是有效的文件或目录")
        return

    # 处理所有图像
    for image_path in image_files:
        print(f"\n正在处理图像: {image_path}")

        # 预处理图像
        print("正在预处理图像...")
        image_tensor, original_size = preprocess_image(image_path, device)

        # 执行推理
        print("正在执行推理...")
        anomaly_map, img_score = simple_infer(model, image_tensor, device, args.use_normalization)

        print(f"图像级异常分数: {img_score:.4f}")

        # 可视化结果
        print("正在可视化结果...")
        visualize_results(image_path, anomaly_map, args.output_dir, args.use_normalization, args.threshold)

    print(f"\n推理完成! 结果保存在 {args.output_dir} 目录中")


if __name__ == '__main__':
    main()