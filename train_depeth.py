from ultralytics.models import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    # 假设 'yolo12s.pt' 是你的模型定义文件对应的预训练权重或模型结构
    model = YOLO('yolo12s.pt') 

    # 根据 RemDet 论文的参数进行训练
    model.train(
        data='/data2/user/2024/lzy/ultralytics12/ultralytics/cfg/datasets/mydataDepth.yml',
        # --- 核心训练参数 ---
        epochs=300,
        batch=8,  # RemDet 总批次为128，这里单卡8
        imgsz=640,
        nbs=128,          # 指定名义批量用于梯度累计与权重衰减缩放，相当于每步累计16次
        
        # --- 优化器相关参数 ---
        optimizer='SGD',  # 明确指定使用SGD
        lr0=0.01,         # RemDet 基准 LR=0.01，配合 accumulate 保持有效学习率
        momentum=0.937,   # 优化器动量
        weight_decay=0.0005, # 权重衰减
        
        # --- 学习率调度器 ---
        cos_lr=True,      # 使用余弦退火学习率调度器，对应论文的 "Flat-Cosine"
        lrf=0.01,         # Ultralytics中 cos_lr=True 时，lrf 设为0.01可模拟更平坦的衰减
        
        # --- 数据增强参数 ---
        mosaic=1.0,       # 启用 Mosaic 增强 (论文明确提到使用)
        mixup=0.2,        # RemDet 补充材料指明 Mosaic + MixUp
        
        # --- 其他参数 ---
        device='4',       # EMA 默认启用，衰减约 0.9999，与论文配置接近
        workers=2,
        cache=False,
        amp=True,
        project='runs/train',
        name='visdrone_aligned_remdet'
    )
    # model.export(format='onnx', imgsz=640, simplify=True, opset=12)