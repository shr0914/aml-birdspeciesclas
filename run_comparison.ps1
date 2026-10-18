# Approach comparison: architecture and pretraining, each with and without regularisation.
# Pass the regularisation preset that won the ablation study (run_ablation.ps1).
# Each run writes history.csv, config.json, best.pt and train.log to runs/<name>/.
#
# Usage (from the repo root):  powershell -ExecutionPolicy Bypass -File run_comparison.ps1 -Reg strong
param([string]$Reg = 'strong')

$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$base = @('--train-dir', 'data/Train', '--train-txt', 'data/train.txt', '--workers', '0')
# From scratch there are no pretrained weights to protect, so it gets a higher learning rate
# for every layer and more epochs to learn from zero
$scratch = @('--no-pretrained', '--epochs', '30', '--lr', '1e-3', '--head-lr-mult', '1')

$runs = [ordered]@{
    'F_convnext_in1k' = @('--model', 'convnext_tiny.fb_in1k', '--epochs', '15', '--reg', $Reg)
    'G_resnet50_none' = @('--model', 'resnet50', '--epochs', '15', '--reg', 'none')
    'H_resnet50'      = @('--model', 'resnet50', '--epochs', '15', '--reg', $Reg)
    'I_scratch_none'  = $scratch + @('--reg', 'none')
    'J_scratch_reg'   = $scratch + @('--reg', $Reg)
}

$start = Get-Date
foreach ($name in $runs.Keys) {
    $out = "runs/$name"
    New-Item -ItemType Directory -Force $out | Out-Null
    Write-Host "`n===== $name  ($(Get-Date -Format HH:mm)) =====" -ForegroundColor Cyan
    # cmd does the 2>&1 redirect: in PowerShell 5.1 it would turn Python warnings into red NativeCommandErrors
    cmd /c python -u train.py @base @($runs[$name]) --out-dir $out '2>&1' | Tee-Object -FilePath "$out/train.log"
}
Write-Host "`nAll runs finished in $([int]((Get-Date) - $start).TotalMinutes) min." -ForegroundColor Green
