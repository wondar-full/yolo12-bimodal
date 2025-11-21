@echo off
REM 在lzy-yolo12环境中测试GGFE
call conda activate lzy-yolo12
python test_ggfe_local.py
pause
