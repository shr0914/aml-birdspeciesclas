# Repeats the final model's configuration (run E: ConvNeXt-Tiny, ImageNet-22k, strong preset)
# with two more training seeds, to show how much its result varies from run to run.
# The validation split stays fixed (--split-seed 42), so only the training randomness changes.
# Seed 42 is the original runs/E_strong. Outputs go to runs/E_strong_seed<N>/.
#
# Usage (from the repo root):  powershell -ExecutionPolicy Bypass -File run_seeds.ps1

$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$base = @('--train-dir', 'data/Train', '--train-txt', 'data/train.txt', '--workers', '0',
          '--epochs', '15', '--reg', 'strong', '--split-seed', '42')

$start = Get-Date
foreach ($seed in 1, 2) {
    $out = "runs/E_strong_seed$seed"
    New-Item -ItemType Directory -Force $out | Out-Null
    Write-Host "`n===== seed $seed  ($(Get-Date -Format HH:mm)) =====" -ForegroundColor Cyan
    # cmd does the 2>&1 redirect: in PowerShell 5.1 it would turn Python warnings into red NativeCommandErrors
    cmd /c python -u train.py @base --seed $seed --out-dir $out '2>&1' | Tee-Object -FilePath "$out/train.log"
}
Write-Host "`nAll runs finished in $([int]((Get-Date) - $start).TotalMinutes) min." -ForegroundColor Green
