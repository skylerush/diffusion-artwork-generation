# Robustly extract the v2 FID reference set, RESUMING after network failures.
#
# Why this wrapper exists: HuggingFace `datasets` streaming has no resume. A single DNS/timeout blip
# kills the stream, and huggingface_hub then *closes its HTTP client*, so even its own retries fail
# ("RuntimeError: Cannot send a request, as the client has been closed"). Our first attempt died
# exactly that way, after 0 images — 50 minutes of streaming is a long window for one hiccup.
#
# Strategy: retry in a FRESH PROCESS (discarding the broken client) and use --skip / --name-offset to
# continue from whatever is already on disk. Parquet shards are cached, so re-scanning the skipped
# rows is fast on later attempts.
$ErrorActionPreference = 'Continue'
$dir = "C:\Users\sagiz\Desktop\Shenkar\neural networks\diffusion-artwork-generation"
$py  = "$dir\.venv\Scripts\python.exe"
$out = "$dir\data\impressionism_512_ref"
$refdir = "$out\heldout"
$log = "$dir\experiments\extract_ref.log"

$TARGET   = 2500   # + the 300 original held-out = 2800 reference images (>= 2048 -> full-rank covariance)
$CONSUMED = 1500   # images v1 already consumed (300 held-out + 1200 train) -> keeps the sets DISJOINT

function Log($m) { ("{0}  {1}" -f (Get-Date).ToString('HH:mm:ss'), $m) | Out-File $log -Append -Encoding utf8 }
function Have { if (Test-Path $refdir) { @(Get-ChildItem $refdir -Filter *.jpg -EA SilentlyContinue).Count } else { 0 } }

Log "================ reference extraction START (target $TARGET) ================"
for ($i = 1; $i -le 25; $i++) {
    $have = Have
    if ($have -ge $TARGET) { Log "TARGET REACHED: $have images"; break }
    $need = $TARGET - $have
    $skip = $CONSUMED + $have          # skip v1's images AND everything we already fetched
    Log "attempt $i : have=$have  need=$need  skip=$skip"
    & $py -u "$dir\src\phase2_sd_finetune\prepare_data.py" `
        --skip $skip --holdout $need --max-images 0 --name-offset $have --out $out 2>&1 |
        Out-File $log -Append -Encoding utf8
    Log "attempt $i ended (exit $LASTEXITCODE, now have $(Have))"
    if ((Have) -lt $TARGET) { Start-Sleep -Seconds 15 }   # let a transient network issue clear
}
Log "================ reference extraction DONE: $(Have) images ================"
