# EVAL v2 — GAP-CLOSING leg. The 6-model v2 sweep (2026-07-18 15:33) left two models unmeasured on
# the honest ruler: lora_r64 and dreambooth. Their claims in the report ("rank saturates", "r64
# drifts", "DreamBooth lowest CLIP") still rested on v1 FIDs we PROVED unreliable (floor 156.7).
#
# It also fixes a flaw v1 never caught: v1 scored DreamBooth on prompts that never contain "sks" —
# i.e. with the method's entire mechanism switched off. We therefore run DreamBooth BOTH ways:
#   dreambooth      : neutral prompts, no trigger  -> apples-to-apples with every other model
#   dreambooth_sks  : neutral prompts + ", in sks style" -> the method as designed
# The suffix deliberately omits the English word "impressionist" (which is in the training instance
# prompt) so the comparison is not contaminated by a style word the other models don't get.
#
# Batch 8, not 16 — a batch-16 generation job oversubscribed the 32 GB alongside a busy desktop,
# the driver paged, throughput fell ~65x and the job wedged for 5h45m before exit -1 (see D10).
#
# RESUMABLE: any model directory already holding 2048 jpgs is skipped, so a relaunch after a crash
# or reboot costs nothing for work already done.
$ErrorActionPreference = 'Continue'
$dir = "C:\Users\sagiz\Desktop\Shenkar\neural networks\diffusion-artwork-generation"
$py  = "$dir\.venv\Scripts\python.exe"
$P2  = "$dir\src\phase2_sd_finetune"
$log = "$dir\experiments\eval_v2.log"
$O   = "$dir\outputs\phase2"
$V2  = "$dir\outputs\phase2_eval_v2"
$N   = 2048

function Log($m) { ("{0}  {1}" -f (Get-Date).ToString('HH:mm:ss'), $m) | Out-File $log -Append -Encoding utf8 }
function Run($label, [string[]]$cmd) {
    Log "BEGIN $label"
    & $py -u @cmd 2>&1 | Out-File $log -Append -Encoding utf8
    Log "END   $label (exit $LASTEXITCODE)"
}
function Gen($name, [string[]]$extra) {
    $d = "$V2\$name\eval_samples"
    $have = @(Get-ChildItem $d -Filter *.jpg -ErrorAction SilentlyContinue).Count
    if ($have -ge $N) { Log "SKIP gen:$name (already $have images)"; return }
    Run "gen:$name" (@("$P2\infer.py", "--n", "$N", "--steps", "25", "--batch", "8", "--neutral",
                       "--out", $d) + $extra)
}

Log "=================== EVAL v2 GAP LEG (r64 + dreambooth x2) ==================="

Gen "lora_r64"       @("--lora", "$O\lora_r64\ckpt\lora_last.pt",    "--rank", "64", "--lora-scale", "1.5")
Gen "dreambooth"     @("--lora", "$O\dreambooth\ckpt\lora_last.pt",  "--rank", "16", "--lora-scale", "1.5")
Gen "dreambooth_sks" @("--lora", "$O\dreambooth\ckpt\lora_last.pt",  "--rank", "16", "--lora-scale", "1.5",
                       "--trigger-suffix", ", in sks style")

Log "=================== GAP GENERATION DONE -> full 9-model eval ==================="

# Final table: all nine models against the same 2,800-image reference and the same measured floor.
Run "eval:v2-full" @("$P2\eval_v2.py",
                "--ref-dirs", "$dir\data\impressionism_512\heldout", "$dir\data\impressionism_512_ref\heldout",
                "--models",
                "base=$V2\base\eval_samples",
                "lora_r4=$V2\lora_r4\eval_samples",
                "lora_r16_s1.0=$V2\lora_r16_s1.0\eval_samples",
                "lora_r16_s1.5=$V2\lora_r16_s1.5\eval_samples",
                "lora_r64=$V2\lora_r64\eval_samples",
                "dreambooth=$V2\dreambooth\eval_samples",
                "dreambooth_sks=$V2\dreambooth_sks\eval_samples",
                "full_ft=$V2\full_ft\eval_samples",
                "full_ft_matched=$V2\full_ft_matched\eval_samples")

Log "======================= EVAL v2 COMPLETE (9 models) ======================="
