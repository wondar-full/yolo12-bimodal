@echo off
REM 批量训练所有尺寸的YOLO12-RGBD模型 (with SOLR loss) - Windows版本
REM 用途: 对比不同模型尺寸 (n/s/m/l/x) 与RemDet的性能差异
REM 
REM 使用方法:
REM   batch_train_solr_all_sizes.bat       # 训练所有尺寸
REM   batch_train_solr_all_sizes.bat n     # 只训练nano
REM   batch_train_solr_all_sizes.bat s     # 只训练small
REM   batch_train_solr_all_sizes.bat m     # 只训练medium

setlocal enabledelayedexpansion

REM ================================================================================================
REM 配置参数
REM ================================================================================================
set DATA_YAML=data/visdrone-rgbd.yaml
set EPOCHS=300
set DEVICE=0
set PROJECT=runs/train_solr

REM SOLR权重配置
set SMALL_WEIGHT=2.5
set MEDIUM_WEIGHT=2.0
set LARGE_WEIGHT=1.0

echo.
echo ================================================================================================
echo 🚀 YOLO12-RGBD Multi-Size Training with SOLR (Windows)
echo ================================================================================================
echo.

REM 检查数据集文件
if not exist "%DATA_YAML%" (
    echo ❌ Dataset config not found: %DATA_YAML%
    echo Please check the path and try again.
    exit /b 1
)
echo ✅ Dataset config found: %DATA_YAML%
echo.

REM 确定要训练的模型尺寸
if "%~1"=="" (
    echo ℹ️  No size specified, will train all sizes: n, s, m, l, x
    echo ℹ️  Estimated total time: ~14-16 hours on RTX 4090
    echo.
    set /p CONFIRM="Continue? [y/N]: "
    if /i not "!CONFIRM!"=="y" (
        echo Training cancelled.
        exit /b 0
    )
    set SIZES=n s m l x
) else (
    echo ℹ️  Will train size: %~1
    set SIZES=%~1
)

echo.
set START_TIME=%TIME%

REM ================================================================================================
REM 训练函数 (通过循环调用)
REM ================================================================================================
for %%s in (%SIZES%) do (
    echo.
    echo ================================================================================================
    echo Training YOLO12-RGBD-%%s with SOLR
    echo ================================================================================================
    echo.
    
    REM 根据模型大小设置batch size
    if "%%s"=="n" set BATCH=32
    if "%%s"=="s" set BATCH=16
    if "%%s"=="m" set BATCH=8
    if "%%s"=="l" set BATCH=4
    if "%%s"=="x" set BATCH=2
    
    REM 设置RemDet对标目标
    if "%%s"=="n" set TARGET=RemDet-Tiny (AP@0.5: 37.1%%, AP_m: 33.0%%)
    if "%%s"=="s" set TARGET=RemDet-S (AP@0.5: 42.3%%, AP_m: 38.5%%)
    if "%%s"=="m" set TARGET=RemDet-M (AP@0.5: 45.0%%, AP_m: 41.2%%)
    if "%%s"=="l" set TARGET=RemDet-L (AP@0.5: 47.4%%, AP_m: 43.6%%)
    if "%%s"=="x" set TARGET=RemDet-X (AP@0.5: 48.3%%, AP_m: 44.8%%)
    
    echo ℹ️  Configuration:
    echo   Model size:    %%s (batch=!BATCH!)
    echo   Target:        !TARGET!
    echo   SOLR weights:  small=%SMALL_WEIGHT%x, medium=%MEDIUM_WEIGHT%x, large=%LARGE_WEIGHT%x
    echo   Epochs:        %EPOCHS%
    echo   Device:        %DEVICE%
    echo   Output:        %PROJECT%/solr_%%s_300ep
    echo.
    
    echo ℹ️  Starting training at %TIME%...
    echo.
    
    REM 执行训练
    python train_depth_solr.py ^
        --data "%DATA_YAML%" ^
        --cfg %%s ^
        --epochs %EPOCHS% ^
        --batch !BATCH! ^
        --device %DEVICE% ^
        --small_weight %SMALL_WEIGHT% ^
        --medium_weight %MEDIUM_WEIGHT% ^
        --large_weight %LARGE_WEIGHT% ^
        --optimizer SGD ^
        --lr0 0.01 ^
        --momentum 0.937 ^
        --weight_decay 0.0005 ^
        --mosaic 1.0 ^
        --mixup 0.15 ^
        --close_mosaic 10 ^
        --amp ^
        --project "%PROJECT%" ^
        --name "solr_%%s_300ep" ^
        --exist_ok
    
    if !errorlevel! equ 0 (
        echo.
        echo ✅ Training completed successfully!
        echo ℹ️  Results saved to: %PROJECT%/solr_%%s_300ep
        echo ℹ️  Finished at %TIME%
        echo.
    ) else (
        echo.
        echo ❌ Training failed for size %%s!
        exit /b 1
    )
    
    REM 训练间隔 (避免GPU过热)
    echo ℹ️  Cooling down for 60 seconds before next training...
    timeout /t 60 /nobreak >nul
)

echo.
echo ================================================================================================
echo 🎉 All Training Completed!
echo ================================================================================================
echo.
echo ℹ️  Start time: %START_TIME%
echo ℹ️  End time:   %TIME%
echo ℹ️  Results directory: %PROJECT%/
echo.

REM 生成结果对比表
echo ================================================================================================
echo 📊 Results Summary
echo ================================================================================================
echo.
echo Model    mAP@0.5      mAP@0.5:0.95 Target (RemDet)
echo --------------------------------------------------------------------------------------------

for %%s in (%SIZES%) do (
    if exist "%PROJECT%\solr_%%s_300ep\results.txt" (
        REM 在Windows上提取结果比较复杂,建议手动查看或使用Python脚本
        echo %%s        See %PROJECT%\solr_%%s_300ep\results.txt
    ) else (
        echo %%s        Training data not found
    )
)

echo.
echo ℹ️  Next steps:
echo   1. Run COCO evaluation: python val_coco_eval.py --weights %PROJECT%/solr_s_300ep/weights/best.pt
echo   2. Compare with RemDet benchmarks
echo   3. Analyze which size achieves best performance/efficiency trade-off
echo.

endlocal
