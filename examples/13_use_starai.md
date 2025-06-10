## 校准
1.零位，爪子打开
2.start_pos 完全打开，后仰
3.end_pos 休息姿态，爪子闭合

python lerobot/scripts/control_robot.py \
--robot.type=starai \
--robot.cameras='{}' \
--control.type=calibrate \
--control.arms='["main_follower"]'

## 摄像头检查
python lerobot/common/robot_devices/cameras/opencv.py \
    --images-dir outputs/images_from_opencv_cameras

## 无摄像头测试
python lerobot/scripts/control_robot.py \
--robot.type=starai \
--robot.cameras='{}' \
--control.type=teleoperate

## 有摄像头测试
python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --control.type=teleoperate \
  --control.display_data=true

## 记录数据
python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=nyancos/starai03 \
  --control.tags='["starai01","tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=20 \
  --control.display_data=true \
  --control.push_to_hub=false

## 训练
  python lerobot/scripts/train.py \
  --dataset.repo_id=nyancos/starai03 \
  --policy.type=act \
  --output_dir=outputs/train/act_starai03 \
  --job_name=act_starai03 \
  --policy.device=cuda \
  --wandb.enable=false

## 评估
python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=nyancos/eval_act_starai03S20001 \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=false \
  --control.policy.path=outputs/train/act_starai03/checkpoints/last/pretrained_model

## 恢复训练
python lerobot/scripts/train.py \
  --config_path=outputs/train/act_starai03/checkpoints/last/pretrained_model/train_config.json \
  --resume=true

python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_follower"]'

sudo apt-get install socat
socat -d -d pty,raw,echo=0 pty,raw,echo=0




#记录训练数据集
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=nyancos/so101_test \
  --control.tags='["so101","tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.display_data=true \
  --control.push_to_hub=false


python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=nyancos/so101_test \
  --control.episode=0
  --control.local_files_only=true

## 训练
  python lerobot/scripts/train.py \
  --dataset.repo_id=nyancos/so101_test \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --wandb.enable=false

  --dataset.local_files_only=false


python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=nyancos/eval_act_so101_test_1 \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/act_so101_test/checkpoints/last/pretrained_model

  watch -n 5 nvidia-smi

python lerobot/scripts/control_robot.py \
--robot.type=so101 \
--robot.cameras='{}' \
--control.type=teleoperate