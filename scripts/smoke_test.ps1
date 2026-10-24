# 全工程を極小設定で通す動作確認（Windows / CPU で数分）
#   .\scripts\smoke_test.ps1
$ErrorActionPreference = "Stop"
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

function Step($name, [string[]]$cmd) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    & $py @cmd
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
}

Remove-Item -Recurse -Force checkpoints\small, data\small, runs\small -ErrorAction SilentlyContinue

Step "1. PPO"        @("-m", "coinrun_dt.ppo.train", "--config", "configs/small/ppo.yaml")
Step "2. collect"    @("-m", "coinrun_dt.data.collect", "--config", "configs/small/collect.yaml")
Step "3. DT"         @("-m", "coinrun_dt.dt.train", "--config", "configs/small/dt.yaml")
Step "4. difficulty" @("-m", "coinrun_dt.difficulty.score", "--config", "configs/small/difficulty.yaml")

foreach ($f in @("data\small\index.csv", "checkpoints\small\dt\dt_final.pt",
                 "runs\small\difficulty\difficulty.csv", "runs\small\difficulty\correlation_spearman.csv")) {
    if (-not (Test-Path $f)) { throw "missing output: $f" }
}
Write-Host "`nsmoke test OK" -ForegroundColor Green
