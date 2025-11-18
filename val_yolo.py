from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/data2/user/2024/lzy/yolo12-bimodal/runs/train/visdrone_x/weights/best.pt')
    model.val(data='/data2/user/2024/lzy/yolo12-bimodal/data/visdrone-rgbd.yaml',
              split='val',
              save_json=True,
              project='/data2/user/2024/lzy/yolo12-bimodal/runs/val/yolo-yolo12x'
              )
