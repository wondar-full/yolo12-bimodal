from ultralytics.models import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    # 用多模态数据集配置，channels=4
    model = YOLO('yolo12s.pt')
    model.train(data='/data2/user/2024/lzy/ultralyticsyolo11/data.yaml', epochs=300, batch=8, device='0', imgsz=640, workers=2, cache=False,
                 amp=True, mosaic=False, project='runs/train', name='visdrone3')
    #model.export(format='onnx', imgsz=640, simplify=True,opset=12)


