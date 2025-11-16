# Phase 3 Quick Upload Script
# 请先修改SERVER_IP为你的服务器地址

$SERVER_IP = "10.16.62.111"  # 修改为你的服务器IP
$SERVER_USER = "ubuntu"
$REMOTE_DIR = "/data2/user/2024/lzy/yolo12-bimodal"

Write-Host "`n=================================" -ForegroundColor Cyan
Write-Host "Phase 3: Quick Upload to Server" -ForegroundColor Cyan
Write-Host "=================================`n" -ForegroundColor Cyan

# File list
$files = @(
    @{local="ultralytics\nn\modules\block.py"; remote="ultralytics/nn/modules/block.py"},
    @{local="ultralytics\cfg\models\12\yolo12s-rgbd-channelc2f.yaml"; remote="ultralytics/cfg/models/12/yolo12s-rgbd-channelc2f.yaml"},
    @{local="train_phase3.py"; remote="train_phase3.py"},
    @{local="test_phase3.py"; remote="test_phase3.py"},
    @{local="verify_phase3.py"; remote="verify_phase3.py"}
)

$success_count = 0
$total_count = $files.Count

foreach ($file in $files) {
    $local_path = "f:\CV\Paper\yoloDepth\yoloDepth\$($file.local)"
    $remote_path = "${SERVER_USER}@${SERVER_IP}:${REMOTE_DIR}/$($file.remote)"
    
    Write-Host "Uploading: $($file.local) ..." -ForegroundColor Yellow
    
    scp $local_path $remote_path
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Success`n" -ForegroundColor Green
        $success_count++
    } else {
        Write-Host "  ❌ Failed`n" -ForegroundColor Red
    }
}

Write-Host "`n=================================" -ForegroundColor Cyan
Write-Host "Upload Summary: $success_count/$total_count files" -ForegroundColor Cyan
Write-Host "=================================`n" -ForegroundColor Cyan

if ($success_count -eq $total_count) {
    Write-Host "✅ All files uploaded successfully!`n" -ForegroundColor Green
    Write-Host "Next steps (on server):" -ForegroundColor Yellow
    Write-Host "  ssh ${SERVER_USER}@${SERVER_IP}" -ForegroundColor White
    Write-Host "  cd $REMOTE_DIR" -ForegroundColor White
    Write-Host "  conda activate lzy-yolo12" -ForegroundColor White
    Write-Host "  python verify_phase3.py" -ForegroundColor White
    Write-Host "  python test_phase3.py" -ForegroundColor White
    Write-Host "  CUDA_VISIBLE_DEVICES=6 nohup python train_phase3.py > train_phase3.log 2>&1 &`n" -ForegroundColor White
} else {
    Write-Host "❌ Some files failed to upload. Please check and retry.`n" -ForegroundColor Red
}
