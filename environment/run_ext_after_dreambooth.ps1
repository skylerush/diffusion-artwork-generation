# Self-sequencing extension: wait for the (externally launched) dreambooth v2 generation to finish,
# then (1) back up the 6-model results, (2) run the FULL 8-model evaluation, (3) render the 8-model
# hero grid for the presentation. Logs to experiments/eval_v2_ext.log.
$ErrorActionPreference = 'Continue'
$dir = "C:\Users\sagiz\Desktop\Shenkar\neural networks\diffusion-artwork-generation"
$py  = "$dir\.venv\Scripts\python.exe"
$P2  = "$dir\src\phase2_sd_finetune"
$log = "$dir\experiments\eval_v2_ext.log"
$V2  = "$dir\outputs\phase2_eval_v2"

function Log($m) { ("{0}  {1}" -f (Get-Date).ToString('HH:mm:ss'), $m) | Out-File $log -Append -Encoding utf8 }
function Run($label, [string[]]$cmd) {
    Log "BEGIN $label"
    & $py -u @cmd 2>&1 | Out-File $log -Append -Encoding utf8
    Log "END   $label (exit $LASTEXITCODE)"
}

Log "=== EXT: waiting for dreambooth generation to finish ==="
$deadline = (Get-Date).AddMinutes(75)
while ((Get-Date) -lt $deadline) {
    $done = Select-String -Path "$dir\experiments\eval_v2.log" -Pattern "END   gen:dreambooth" -Quiet
    $n = @(Get-ChildItem "$V2\dreambooth\eval_samples" -Filter *.jpg -ErrorAction SilentlyContinue).Count
    if ($done -or $n -ge 2048) { break }
    Start-Sleep -Seconds 30
}
Log ("EXT: wait over (dreambooth imgs: " + @(Get-ChildItem "$V2\dreambooth\eval_samples" -Filter *.jpg -ErrorAction SilentlyContinue).Count + ")")

Copy-Item "$dir\experiments\eval_v2_results.json" "$dir\experiments\eval_v2_results_6models.json" -Force

Run "eval:v2-8models" @("$P2\eval_v2.py",
    "--ref-dirs", "$dir\data\impressionism_512\heldout", "$dir\data\impressionism_512_ref\heldout",
    "--models",
    "base=$V2\base\eval_samples",
    "lora_r4=$V2\lora_r4\eval_samples",
    "lora_r16_s1.0=$V2\lora_r16_s1.0\eval_samples",
    "lora_r16_s1.5=$V2\lora_r16_s1.5\eval_samples",
    "lora_r64=$V2\lora_r64\eval_samples",
    "dreambooth=$V2\dreambooth\eval_samples",
    "full_ft=$V2\full_ft\eval_samples",
    "full_ft_matched=$V2\full_ft_matched\eval_samples")

Run "hero-grid" @("$P2\hero_grid.py")
Log "=== EXT ALL DONE ==="
