# Overfitting ablation study: adds regularisation one step at a time.
# Same seed, validation split and epochs for every run, so only regularisation differs.
# Each run writes history.csv, config.json, best.pt and train.log to runs/<name>/.
#
# Usage (from the repo root):  powershell -ExecutionPolicy Bypass -File run_ablation.ps1

$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$base = @('--train-dir', 'data/Train', '--train-txt', 'data/train.txt', '--epochs', '15', '--workers', '0')

$runs = [ordered]@{
    'A_none'    = @('--reg', 'none')
    'B_aug'     = @('--reg', 'none', '--aug', 'basic')
    'C_default' = @('--reg', 'default')
    'D_mixup'   = @('--reg', 'default', '--mixup', '0.8', '--cutmix', '1.0', '--mixup-prob', '0.5')
    'E_strong'  = @('--reg', 'strong')
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
