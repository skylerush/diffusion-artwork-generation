# Unattended experiment sweep. Runs every remaining experiment + eval sequentially,
# continues past any single failure, and logs everything to experiments/sweep.log.
# Launch:  powershell -ExecutionPolicy Bypass -File environment\run_sweep.ps1
$ErrorActionPreference = 'Continue'
$dir = "C:\Users\sagiz\Desktop\Shenkar\neural networks\diffusion-artwork-generation"
$py  = "$dir\.venv\Scripts\python.exe"
$P1  = "$dir\src\phase1_ddpm_from_scratch"
$P2  = "$dir\src\phase2_sd_finetune"
$log = "$dir\experiments\sweep.log"
$N   = "256"          # eval images per model
$U   = "0.65"         # GPU throttle (set to 1.0 for full speed)
$IMP = "$dir\data\impressionism_512\train"
$O   = "$dir\outputs\phase2"

function Log($m) { ("{0}  {1}" -f (Get-Date).ToString('HH:mm:ss'), $m) | Out-File $log -Append -Encoding utf8 }
function Run($label, [string[]]$cmd) {
    Log "BEGIN $label"
    & $py -u @cmd 2>&1 | Out-File $log -Append -Encoding utf8
    Log "END   $label (exit $LASTEXITCODE)"
}
function Infer($name, [string[]]$ckpt) {
    Run "infer:$name" (@("$P2\infer.py", "--run-name", $name, "--n", $N, "--batch", "8") + $ckpt)
}
function Eval($name) {
    Run "eval:$name" @("$P2\eval.py", "--gen-dir", "$O\$name\eval_samples", "--captions", "$O\$name\eval_samples\prompts.jsonl")
}

Log "===================== SWEEP START ====================="

# --- Phase 2: baseline + LoRA rank sweep + full-FT + DreamBooth ---
Infer "base" @()
Eval  "base"

Infer "lora_r16" @("--lora", "$O\lora_r16\ckpt\lora_last.pt", "--rank", "16")
Eval  "lora_r16"

Run   "train:lora_r4" @("$P2\train_lora.py", "--rank", "4", "--steps", "1500", "--run-name", "lora_r4", "--max-util", $U)
Infer "lora_r4" @("--lora", "$O\lora_r4\ckpt\lora_last.pt", "--rank", "4")
Eval  "lora_r4"

Run   "train:lora_r64" @("$P2\train_lora.py", "--rank", "64", "--steps", "1500", "--run-name", "lora_r64", "--max-util", $U)
Infer "lora_r64" @("--lora", "$O\lora_r64\ckpt\lora_last.pt", "--rank", "64")
Eval  "lora_r64"

Run   "train:full_ft" @("$P2\train_full.py", "--steps", "1500", "--run-name", "full_ft", "--max-util", $U)
Infer "full_ft" @("--unet", "$O\full_ft\ckpt\unet_last")
Eval  "full_ft"

Run   "train:dreambooth" @("$P2\train_dreambooth.py", "--with-prior", "--steps", "1200", "--run-name", "dreambooth", "--max-util", $U)
Infer "dreambooth" @("--lora", "$O\dreambooth\ckpt\lora_last.pt", "--rank", "16")
Eval  "dreambooth"

# --- Phase 1 extras: failure-ladder (lr too high) + Impressionism-64 ---
Run "train:p1_r01_lr2e3" @("$P1\train.py", "--data", "butterflies", "--run-name", "p1_r01_lr2e3", "--lr", "0.002", "--steps", "2000", "--sample-every", "500", "--ckpt-every", "1000", "--max-util", $U)
Run "train:p1b_impressionism" @("$P1\train.py", "--data", $IMP, "--image-size", "64", "--run-name", "p1b_impressionism", "--steps", "10000", "--sample-every", "2000", "--ckpt-every", "2000", "--max-util", $U)

Log "===================== SWEEP DONE ====================="
# BUG FIX: never pipe Select-String of $log back into $log — reading and appending the same file
# self-feeds and grows without bound (this once produced a 177 GB log). Materialise the matches
# into a variable first and write them to a SEPARATE file.
$summary = @(Select-String -Path $log -Pattern "BEGIN |FID = |CLIPScore = |exit " | ForEach-Object { $_.Line })
$summary | Out-File (Join-Path (Split-Path $log) "sweep_summary.txt") -Encoding utf8
Log "summary -> experiments\sweep_summary.txt ($($summary.Count) lines)"
