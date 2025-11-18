python train_uav_joint.py --name uavdt_n --data /data2/user/2024/lzy/yolo12-bimodal/data/uavdt-rgbd.yaml --device 4 --pretrained /data2/user/2024/lzy/yolo12-bimodal/models/yolo12n.pt --cfg n --batch 16 --epochs 300
python train_uav_joint.py --name uavdt_m --data /data2/user/2024/lzy/yolo12-bimodal/data/uavdt-rgbd.yaml --device 5 --pretrained /data2/user/2024/lzy/yolo12-bimodal/models/yolo12m.pt --cfg m --batch 16 --epochs 300
python train_uav_joint.py --name uavdt_l --data /data2/user/2024/lzy/yolo12-bimodal/data/uavdt-rgbd.yaml --device 6 --pretrained /data2/user/2024/lzy/yolo12-bimodal/models/yolo12l.pt --cfg l --batch 16 --epochs 300
python train_uav_joint.py --name uavdt_x --data /data2/user/2024/lzy/yolo12-bimodal/data/uavdt-rgbd.yaml --device 7 --pretrained /data2/user/2024/lzy/yolo12-bimodal/models/yolo12x.pt --cfg x --batch 8 --epochs 300



python train_uav_joint.py --name uavdt_x --data /data2/user/2024/lzy/yolo12-bimodal/data/visdrone-rgbd.yaml --device 7 --pretrained /data2/user/2024/lzy/yolo12-bimodal/models/yolo12x.pt --cfg x --batch 8 --epochs 300




python train_uav_joint.py --name visdrone_n --data /data2/user/2024/lzy/yolo12-bimodal/data/visdrone-rgbd.yaml --device 0 --pretrained /data2/user/2024/lzy/yolo12-bimodal/models/yolo12n.pt --cfg n --batch 16 --epochs 300
python train_uav_joint.py --name visdrone_m --data /data2/user/2024/lzy/yolo12-bimodal/data/visdrone-rgbd.yaml --device 1 --pretrained /data2/user/2024/lzy/yolo12-bimodal/models/yolo12m.pt --cfg m --batch 16 --epochs 300
python train_uav_joint.py --name visdrone_l --data /data2/user/2024/lzy/yolo12-bimodal/data/visdrone-rgbd.yaml --device 2 --pretrained /data2/user/2024/lzy/yolo12-bimodal/models/yolo12l.pt --cfg l --batch 16 --epochs 300
python train_uav_joint.py --name visdrone_x --data /data2/user/2024/lzy/yolo12-bimodal/data/visdrone-rgbd.yaml --device 3 --pretrained /data2/user/2024/lzy/yolo12-bimodal/models/yolo12x.pt --cfg x --batch 8 --epochs 300




python train_depth.py --name rgbd_v2.1_joint_300ep_n --device 1 --data /data2/user/2024/lzy/yolo12-bimodal/data/uav-joint-rgbd.yaml --weights /data2/user/2024/lzy/yolo12-bimodal/models/yolo12n.pt --batch 16 --epochs 300
python train_depth.py --name rgbd_v2.1_joint_300ep_m --device 2 --data /data2/user/2024/lzy/yolo12-bimodal/data/uav-joint-rgbd.yaml --weights /data2/user/2024/lzy/yolo12-bimodal/models/yolo12m.pt --batch 16 --epochs 300
python train_depth.py --name rgbd_v2.1_joint_300ep_l --device 3 --data /data2/user/2024/lzy/yolo12-bimodal/data/uav-joint-rgbd.yaml --weights /data2/user/2024/lzy/yolo12-bimodal/models/yolo12l.pt --batch 8 --epochs 300
python train_depth.py --name rgbd_v2.1_joint_300ep_x --device 4 --data /data2/user/2024/lzy/yolo12-bimodal/data/uav-joint-rgbd.yaml --weights /data2/user/2024/lzy/yolo12-bimodal/models/yolo12x.pt --batch 8 --epochs 300


# 5. COCO评估测试
python val_coco_eval.py --name visdrone_coco_eval_n --weights /data2/user/2024/lzy/yolo12-bimodal/runs/train/visdrone_n/weights/best.pt --data /data2/user/2024/lzy/yolo12-bimodal/data/visdrone-rgbd.yaml
python val_coco_eval.py --name visdrone_coco_eval_m --weights /data2/user/2024/lzy/yolo12-bimodal/runs/train/visdrone_m/weights/best.pt --data /data2/user/2024/lzy/yolo12-bimodal/data/visdrone-rgbd.yaml
python val_coco_eval.py --name visdrone_coco_eval_l --weights /data2/user/2024/lzy/yolo12-bimodal/runs/train/visdrone_l/weights/best.pt --data /data2/user/2024/lzy/yolo12-bimodal/data/visdrone-rgbd.yaml
python val_coco_eval.py --name visdrone_coco_eval_x --weights /data2/user/2024/lzy/yolo12-bimodal/runs/train/visdrone_x/weights/best.pt --data /data2/user/2024/lzy/yolo12-bimodal/data/visdrone-rgbd.yaml

python val_coco_eval.py --name uavdt_coco_eval_n --weights runs/.../best.pt --data data/uavdt-rgbd.yaml
python val_coco_eval.py --name uavdt_coco_eval_m --weights runs/.../best.pt --data data/uavdt-rgbd.yaml
python val_coco_eval.py --name uavdt_coco_eval_l --weights runs/.../best.pt --data data/uavdt-rgbd.yaml
python val_coco_eval.py --name uavdt_coco_eval_x --weights runs/.../best.pt --data data/uavdt-rgbd.yaml


# 6. 完整训练 (300 epochs)
nohup python train_uav_joint.py --data data/visdrone-rgbd.yaml --cfg n --epochs 300 
nohup python train_uav_joint.py --data data/uavdt-rgbd.yaml --cfg n --epochs 300 



