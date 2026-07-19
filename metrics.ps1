# Show all project metrics tables.
#   From a terminal:   .\metrics.ps1
#   Right-click -> Run with PowerShell also works: the window PAUSES at the end instead of
#   closing, and a copy of the output is always saved to  experiments\metrics_latest.txt
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\src\common\show_metrics.py" |
    Tee-Object -FilePath "$PSScriptRoot\experiments\metrics_latest.txt"
Write-Host ""
Write-Host "(saved a copy to experiments\metrics_latest.txt)"
if (-not [Console]::IsInputRedirected) {
    Read-Host "Press Enter to close" | Out-Null
}
