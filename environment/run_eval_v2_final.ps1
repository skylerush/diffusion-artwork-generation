# EVAL v2 — FINAL leg. History: original sweep wedged on VRAM oversubscription (fixed, batch 16->8);
# the resume then completed s1.5 + full_ft but was killed at 00:35 by a machine REBOOT mid
# full_ft_matched (1056/2048). This script finishes the remaining work:
#   regen full_ft_matched (infer.py clears the partial folder) -> regen lora_r16_s1.0 -> eval.
$ErrorActionPreference = 'Continue'
$dir = "C:\Users\sagiz\Desktop\Shenkar\neural networks\diffusion-artwork-generation"
$py  = "$dir\.venv\Scripts\python.exe"
$P2  = "$dir\src\phase2_sd_finetune"
$log = "$dir\experiments\eval_v2.log"
$O   = "$dir\outputs\phase2"
$V2  = "$dir\outputs\phase2_eval_v2"

function Log($m) { ("{0}  {1}" -f (Get-Date).ToString('HH:mm:ss'), $m) | Out-File $log -Append -Encoding utf8 }
function Run($label, [string[]]$cmd) {
    Log "BEGIN $label"
    & $py -u @cmd 2>&1 | Out-File $log -Append -Encoding utf8
    Log "END   $label (exit $LASTEXITCODE)"
}
function Gen($name, [string[]]$extra) {
    Run "gen:$name" (@("$P2\infer.py", "--n", "2048", "--steps", "25", "--batch", "8", "--neutral",
                       "--out", "$V2\$name\eval_samples") + $extra)
}

Log "=================== EVAL v2 FINAL LEG (post-reboot) ==================="
Gen "full_ft_matched" @("--unet", "$O\full_ft_matched\ckpt\unet_last")
Gen "lora_r16_s1.0"   @("--lora", "$O\lora_r16\ckpt\lora_last.pt", "--rank", "16", "--lora-scale", "1.0")
Run "eval:v2" @("$P2\eval_v2.py",
                "--ref-dirs", "$dir\data\impressionism_512\heldout", "$dir\data\impressionism_512_ref\heldout",
                "--models",
                "base=$V2\base\eval_samples",
                "lora_r4=$V2\lora_r4\eval_samples",
                "lora_r16_s1.0=$V2\lora_r16_s1.0\eval_samples",
                "lora_r16_s1.5=$V2\lora_r16_s1.5\eval_samples",
                "full_ft=$V2\full_ft\eval_samples",
                "full_ft_matched=$V2\full_ft_matched\eval_samples")
Log "======================= EVAL v2 ALL DONE ======================="
