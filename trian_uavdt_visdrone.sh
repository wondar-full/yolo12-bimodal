python train_uav_joint.py --name rgbd_v2.1_joint_300ep_n --device 5 --pretrained /data2/user/2024/lzy/yolo12-bimodal/models/yolo12n.pt --cfg n --batch 16 --epochs 300
python train_uav_joint.py --name rgbd_v2.1_joint_300ep_m --device 6 --pretrained /data2/user/2024/lzy/yolo12-bimodal/models/yolo12m.pt --cfg m --batch 16 --epochs 300
python train_uav_joint.py --name rgbd_v2.1_joint_300ep_l --device 7 --pretrained /data2/user/2024/lzy/yolo12-bimodal/models/yolo12l.pt --cfg l --batch 2 --epochs 300
python train_uav_joint.py --name rgbd_v2.1_joint_300ep_x --device 0 --pretrained /data2/user/2024/lzy/yolo12-bimodal/models/yolo12x.pt --cfg x --batch 8 --epochs 300




python train_depth.py --name rgbd_v2.1_joint_300ep_n --device 1 --data /data2/user/2024/lzy/yolo12-bimodal/data/uav-joint-rgbd.yaml --weights /data2/user/2024/lzy/yolo12-bimodal/models/yolo12n.pt --batch 16 --epochs 300
python train_depth.py --name rgbd_v2.1_joint_300ep_m --device 2 --data /data2/user/2024/lzy/yolo12-bimodal/data/uav-joint-rgbd.yaml --weights /data2/user/2024/lzy/yolo12-bimodal/models/yolo12m.pt --batch 16 --epochs 300
python train_depth.py --name rgbd_v2.1_joint_300ep_l --device 3 --data /data2/user/2024/lzy/yolo12-bimodal/data/uav-joint-rgbd.yaml --weights /data2/user/2024/lzy/yolo12-bimodal/models/yolo12l.pt --batch 8 --epochs 300
python train_depth.py --name rgbd_v2.1_joint_300ep_x --device 4 --data /data2/user/2024/lzy/yolo12-bimodal/data/uav-joint-rgbd.yaml --weights /data2/user/2024/lzy/yolo12-bimodal/models/yolo12x.pt --batch 8 --epochs 300
