# EVALUATION v2 — fixes the three flaws in our original measurement (see JOURNEY.md).
#   #1  ruler       : 2048 generated images per model (v1 used 256 -> rank-deficient covariance)
#   #2  LoRA scale  : use the tuned inference scale 1.5 (v1 used 1.0 and under-applied our own model)
#   #4  prompts     : NEUTRAL (no "impressionist", no artist) -> isolates what the fine-tune adds
#   #5  confound    : retrain full-FT with images-seen MATCHED to LoRA (8 img/step, 1500 steps)
#
# NOTE: the summary is written to a SEPARATE file. Never Select-String a log and Out-File back into
# the SAME log — that self-feeds and once produced a 177 GB file.
$ErrorActionPreference = 'Continue'
$dir = "C:\Users\sagiz\Desktop\Shenkar\neural networks\diffusion-artwork-generation"
$py  = "$dir\.venv\Scripts\python.exe"
$P2  = "$dir\src\phase2_sd_finetune"
$log = "$dir\experiments\eval_v2.log"
$O   = "$dir\outputs\phase2"
$V2  = "$dir\outputs\phase2_eval_v2"
$N     = "2048"   # >= 2048 so the 2048-dim Inception covariance is full-rank
$STEPS = "25"
$BATCH = "16"

function Log($m) { ("{0}  {1}" -f (Get-Date).ToString('HH:mm:ss'), $m) | Out-File $log -Append -Encoding utf8 }
function Run($label, [string[]]$cmd) {
    Log "BEGIN $label"
    & $py -u @cmd 2>&1 | Out-File $log -Append -Encoding utf8
    Log "END   $label (exit $LASTEXITCODE)"
}
function Gen($name, [string[]]$extra) {
    Run "gen:$name" (@("$P2\infer.py", "--n", $N, "--steps", $STEPS, "--batch", $BATCH, "--neutral",
                       "--out", "$V2\$name\eval_samples") + $extra)
}

Log "======================= EVAL v2 START ======================="

# ---- #5: full fine-tune, images-seen MATCHED to LoRA (batch 2 x accum 4 = 8 img/step) ----
Run "train:full_ft_matched" @("$P2\train_full.py", "--steps", "1500", "--batch-size", "2",
                              "--grad-accum", "4", "--run-name", "full_ft_matched")

# ---- #1 + #2 + #4: 2048 neutral-prompt images per model ----
Gen "base"            @()
Gen "lora_r4"         @("--lora", "$O\lora_r4\ckpt\lora_last.pt",  "--rank", "4",  "--lora-scale", "1.5")
Gen "lora_r16_s1.0"   @("--lora", "$O\lora_r16\ckpt\lora_last.pt", "--rank", "16", "--lora-scale", "1.0")
Gen "lora_r16_s1.5"   @("--lora", "$O\lora_r16\ckpt\lora_last.pt", "--rank", "16", "--lora-scale", "1.5")
Gen "full_ft"         @("--unet", "$O\full_ft\ckpt\unet_last")
Gen "full_ft_matched" @("--unet", "$O\full_ft_matched\ckpt\unet_last")

Log "======================= EVAL v2 GENERATION DONE ======================="

# ---- wait for the reference extraction (separate detached job), then run the final evaluation ----
$refDir1 = "$dir\data\impressionism_512\heldout"        # 300 originals (v1 held-out, never trained on)
$refDir2 = "$dir\data\impressionism_512_ref\heldout"    # ~2500 fresh, disjoint from training
$deadline = (Get-Date).AddHours(3)
while ((Get-Date) -lt $deadline) {
    $n = @(Get-ChildItem $refDir2 -Filter *.jpg -ErrorAction SilentlyContinue).Count
    if ($n -ge 2500) { break }
    $tail = Get-Content "$dir\experiments\extract_ref.log" -ErrorAction SilentlyContinue | Select-Object -Last 1
    if ($tail -match "extraction DONE") { break }
    Start-Sleep -Seconds 60
}
$nref = @(Get-ChildItem $refDir2 -Filter *.jpg -ErrorAction SilentlyContinue).Count
Log "reference wait over: $nref fresh + 300 original reference images"

Run "eval:v2" @("$P2\eval_v2.py",
                "--ref-dirs", $refDir1, $refDir2,
                "--models",
                "base=$V2\base\eval_samples",
                "lora_r4=$V2\lora_r4\eval_samples",
                "lora_r16_s1.0=$V2\lora_r16_s1.0\eval_samples",
                "lora_r16_s1.5=$V2\lora_r16_s1.5\eval_samples",
                "full_ft=$V2\full_ft\eval_samples",
                "full_ft_matched=$V2\full_ft_matched\eval_samples")
Log "======================= EVAL v2 ALL DONE ======================="
