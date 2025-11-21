@echo off
REM ========================================
REM GGFE模块上传脚本 - Windows版本
REM ========================================

echo ========================================
echo 准备上传GGFE模块到服务器
echo ========================================
echo.

REM 设置本地路径
set LOCAL_BASE=f:\CV\Paper\yoloDepth\yolo12-bimodal
set REMOTE_BASE=/data2/user/2024/lzy/yolo12-bimodal

REM 需要上传的文件列表
echo 将上传以下文件:
echo 1. ultralytics\nn\modules\ggfe.py
echo 2. ultralytics\nn\modules\rgbd_ggfe_fusion.py
echo 3. ultralytics\nn\modules\__init__.py
echo 4. check_rgbd_ggfe_fusion_exists.sh
echo.

echo ========================================
echo 请选择上传方式:
echo ========================================
echo 1. 使用 WinSCP (推荐)
echo 2. 使用 pscp (PuTTY)
echo 3. 手动复制文件路径
echo 4. 退出
echo.

set /p choice=请输入选项 (1-4): 

if "%choice%"=="1" goto winscp
if "%choice%"=="2" goto pscp
if "%choice%"=="3" goto manual
if "%choice%"=="4" goto end

:winscp
echo.
echo ========================================
echo WinSCP 命令
echo ========================================
echo 请在 WinSCP 中执行以下操作:
echo.
echo 1. 连接到服务器
echo 2. 导航到远程目录: %REMOTE_BASE%
echo 3. 上传以下文件:
echo.
echo    本地: %LOCAL_BASE%\ultralytics\nn\modules\ggfe.py
echo    远程: %REMOTE_BASE%/ultralytics/nn/modules/ggfe.py
echo.
echo    本地: %LOCAL_BASE%\ultralytics\nn\modules\rgbd_ggfe_fusion.py
echo    远程: %REMOTE_BASE%/ultralytics/nn/modules/rgbd_ggfe_fusion.py
echo.
echo    本地: %LOCAL_BASE%\ultralytics\nn\modules\__init__.py
echo    远程: %REMOTE_BASE%/ultralytics/nn/modules/__init__.py
echo.
echo    本地: %LOCAL_BASE%\check_rgbd_ggfe_fusion_exists.sh
echo    远程: %REMOTE_BASE%/check_rgbd_ggfe_fusion_exists.sh
echo.
pause
goto end

:pscp
echo.
echo ========================================
echo PSCP (PuTTY) 命令
echo ========================================
echo 请将以下命令替换 USER@SERVER 后复制执行:
echo.
echo pscp "%LOCAL_BASE%\ultralytics\nn\modules\ggfe.py" USER@SERVER:%REMOTE_BASE%/ultralytics/nn/modules/
echo.
echo pscp "%LOCAL_BASE%\ultralytics\nn\modules\rgbd_ggfe_fusion.py" USER@SERVER:%REMOTE_BASE%/ultralytics/nn/modules/
echo.
echo pscp "%LOCAL_BASE%\ultralytics\nn\modules\__init__.py" USER@SERVER:%REMOTE_BASE%/ultralytics/nn/modules/
echo.
echo pscp "%LOCAL_BASE%\check_rgbd_ggfe_fusion_exists.sh" USER@SERVER:%REMOTE_BASE%/
echo.
pause
goto end

:manual
echo.
echo ========================================
echo 手动上传路径列表
echo ========================================
echo.
echo 【文件1】
echo 本地: %LOCAL_BASE%\ultralytics\nn\modules\ggfe.py
echo 远程: %REMOTE_BASE%/ultralytics/nn/modules/ggfe.py
echo.
echo 【文件2】
echo 本地: %LOCAL_BASE%\ultralytics\nn\modules\rgbd_ggfe_fusion.py
echo 远程: %REMOTE_BASE%/ultralytics/nn/modules/rgbd_ggfe_fusion.py
echo.
echo 【文件3】
echo 本地: %LOCAL_BASE%\ultralytics\nn\modules\__init__.py
echo 远程: %REMOTE_BASE%/ultralytics/nn/modules/__init__.py
echo.
echo 【文件4】
echo 本地: %LOCAL_BASE%\check_rgbd_ggfe_fusion_exists.sh
echo 远程: %REMOTE_BASE%/check_rgbd_ggfe_fusion_exists.sh
echo.
echo ========================================
echo 上传后验证命令 (在服务器执行):
echo ========================================
echo.
echo cd /data2/user/2024/lzy/yolo12-bimodal
echo chmod +x check_rgbd_ggfe_fusion_exists.sh
echo bash check_rgbd_ggfe_fusion_exists.sh
echo.
echo 预期输出: ✅ All modules imported successfully
echo.
pause
goto end

:end
echo.
echo ========================================
echo 完成
echo ========================================
pause
